## Building a Podcast Clipping Agent Pipeline in Python

A practical guide to building an automated pipeline that takes a raw podcast episode (audio or video) and outputs short, shareable clips — the kind you'd post to YouTube Shorts, TikTok, or Reels.

**Assumptions made for this guide** (adjust as needed):

*   Input is a local audio/video file (mp3/wav/mp4). YouTube URL ingestion can be added with `yt-dlp`.
*   You want vertical, captioned clips ready for social platforms.
*   You're using an LLM (Claude or similar) to decide _which_ moments are clip-worthy, rather than hand-written heuristics.
*   Single-machine pipeline to start; notes on scaling are at the end.

## 1\. Pipeline Overview

```plaintext
 ┌───────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────┐
 │  Ingest   │→ │ Transcribe    │→ │ Score/Select  │→ │  Cut + Render  │→ │  Post-process │
 │ (download │  │ (Whisper +    │  │ Highlights    │  │  Clips         │  │  (captions,   │
 │  / load)  │  │  timestamps)  │  │ (LLM agent)   │  │  (ffmpeg)      │  │  crop, export)│
 └───────────┘   └──────────────┘   └───────────────┘   └────────────────┘   └──────────────┘
```

Each stage is a separate, testable Python module. The "agent" part sits in stage 3 (and optionally stage 5), where an LLM reasons over the transcript rather than following fixed rules.

## 2\. Recommended Stack

| Stage | Library | Notes |
| --- | --- | --- |
| Ingest | `yt-dlp`, `ffmpeg-python` | Download/normalize audio |
| Transcription | `faster-whisper` | Fast, word-level timestamps, runs locally |
| Highlight scoring | Anthropic API (`anthropic` SDK) | Reasons over transcript chunks |
| Clip cutting | `ffmpeg` (via subprocess) | Frame-accurate, fast |
| Captions | `ffmpeg` subtitle burn-in, or `moviepy` | Word-by-word "karaoke" captions are popular for shorts |
| Orchestration | Plain Python + `asyncio`, or Prefect/LangGraph for larger scale | Start simple |

Project setup and dependency management with `uv`:

```bash
uv init
uv add faster-whisper anthropic ffmpeg-python yt-dlp pydub
```

You'll also need the `ffmpeg` binary itself available on PATH.

## 3\. Stage-by-Stage Implementation

### 3.1 Ingest

```python
# ingest.py
import subprocess
from pathlib import Path

def normalize_audio(input_path: str, out_dir: str = "work") -> str:
    """Extract mono 16kHz WAV audio — the format Whisper wants."""
    Path(out_dir).mkdir(exist_ok=True)
    out_path = f"{out_dir}/audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", out_path
    ], check=True, capture_output=True)
    return out_path
```

For YouTube sources, swap the input for a `yt-dlp` download step first.

### 3.2 Transcription (with timestamps)

Word-level timestamps are essential — they're what let you cut clips precisely later.

```python
# transcribe.py
from faster_whisper import WhisperModel

def transcribe(audio_path: str, model_size: str = "medium"):
    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    transcript = []
    for seg in segments:
        transcript.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end}
                      for w in (seg.words or [])]
        })
    return transcript
```

Save this to JSON — every downstream stage reads from it.

### 3.3 Highlight Selection (the "agent" step)

This is where an LLM adds real value over keyword heuristics: it can judge narrative arcs, punchlines, emotional beats, and standalone comprehensibility (a clip needs to make sense without the rest of the episode).

```python
# select_highlights.py
import json
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You select short clips (30-90s) from a podcast transcript
that would perform well as standalone social media clips. Good clips:
- Make sense without additional context
- Have a clear hook in the first 3 seconds
- Contain a complete thought, story, or punchline
- Avoid mid-sentence cuts

Given a transcript with timestamps, return a JSON array of candidate clips:
[{"start": float, "end": float, "reason": str, "hook": str}]
Return ONLY the JSON array, nothing else."""

def select_highlights(transcript: list[dict], max_clips: int = 8) -> list[dict]:
    # Chunk long transcripts to stay within context; here we assume it fits.
    full_text = "\n".join(
        f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}"
        for seg in transcript
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content":
            f"Transcript:\n{full_text}\n\nReturn up to {max_clips} clips."}]
    )

    raw = response.content[0].text.strip()
    return json.loads(raw)
```

**Notes on scaling this step:**

*   For episodes over ~30-45 min, chunk the transcript (e.g. 10-minute windows with overlap) and score each chunk, then merge/rank results.
*   Ask the model to score each candidate (1-10) on "hook strength" and "standalone clarity" so you can filter to your best N.
*   Cache results — re-running the LLM call on the same transcript should be avoided; store `transcript_hash -> highlights` in a local SQLite/JSON store.

### 3.4 Cutting Clips

```python
# cut_clips.py
import subprocess
from pathlib import Path

def cut_clip(source_video: str, start: float, end: float, out_path: str):
    duration = end - start
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", source_video,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        out_path
    ], check=True, capture_output=True)

def cut_all(source_video: str, clips: list[dict], out_dir: str = "clips"):
    Path(out_dir).mkdir(exist_ok=True)
    paths = []
    for i, clip in enumerate(clips):
        out_path = f"{out_dir}/clip_{i:02d}.mp4"
        cut_clip(source_video, clip["start"], clip["end"], out_path)
        paths.append(out_path)
    return paths
```

### 3.5 Vertical Crop + Burned-in Captions

```python
# post_process.py
import subprocess

def to_vertical_with_captions(clip_path: str, srt_path: str, out_path: str):
    """Crop to 9:16 and burn in subtitles for Shorts/Reels/TikTok."""
    vf = (
        "crop=ih*9/16:ih,"          # center crop to 9:16
        f"subtitles={srt_path}:force_style="
        "'FontName=Arial,FontSize=14,PrimaryColour=&HFFFFFF&,"
        "OutlineColour=&H000000&,BorderStyle=1,Outline=2'"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", vf,
        "-c:a", "copy",
        out_path
    ], check=True, capture_output=True)
```

Generate the `.srt` from the word-level timestamps you already have from Whisper (map each clip's word list to relative timestamps within the clip).

## 4\. Orchestration

For a single-episode, single-machine pipeline, a plain script is enough:

```python
# pipeline.py
from ingest import normalize_audio
from transcribe import transcribe
from select_highlights import select_highlights
from cut_clips import cut_all
from post_process import to_vertical_with_captions

def run_pipeline(input_video: str):
    audio = normalize_audio(input_video)
    transcript = transcribe(audio)
    highlights = select_highlights(transcript)
    clip_paths = cut_all(input_video, highlights)

    final_paths = []
    for path, clip in zip(clip_paths, highlights):
        srt = build_srt_for_clip(transcript, clip)   # helper, map words -> srt
        out = path.replace(".mp4", "_final.mp4")
        to_vertical_with_captions(path, srt, out)
        final_paths.append(out)

    return final_paths

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])
```

Run the pipeline using `uv`:

```bash
uv run python pipeline.py input_video.mp4
```

**When to graduate to a real orchestrator** (Prefect, Dagster, or LangGraph if you want the LLM to make multi-step decisions, e.g. re-scoring clips or retrying transcription on low-confidence segments):

*   You're processing many episodes on a schedule
*   You need retries, caching, and observability per stage
*   The "agent" needs to loop (e.g. self-critique a clip selection before finalizing) rather than run once straight through

A minimal LangGraph version would replace `select_highlights` with a graph node that can loop between "propose clips" → "critique clips" → "revise" before passing to the cutting stage.

## 5\. Practical Tips

*   **Keep raw transcript + timestamps as your source of truth.** Every stage should be re-runnable from that JSON without re-transcribing.
*   **Filter for standalone clarity.** The single biggest failure mode in auto-generated clips is context-dependence — a clip that only makes sense if you heard the previous 10 minutes. Have the LLM explicitly score this.
*   **Batch the LLM call.** Score the whole transcript in one or a few calls rather than per-segment; it's cheaper and gives the model narrative context.
*   **Validate clip length.** Enforce a hard min/max (e.g. 20-90s) in code after the LLM proposes candidates — don't rely on the prompt alone.
*   **Log everything.** Store input file, transcript hash, LLM prompt/response, and final clip metadata together so you can debug or reprocess later.

## 6\. Suggested Repo Structure

```plaintext
podcast-clipper/
├── pyproject.toml
├── uv.lock
├── ingest.py
├── transcribe.py
├── select_highlights.py
├── cut_clips.py
├── post_process.py
├── pipeline.py
├── work/          # intermediate audio, transcripts
├── clips/         # raw cut clips
└── output/        # final vertical+captioned clips
```

## Next Steps

*   Add a `yt-dlp` ingest step if you're pulling from YouTube directly.
*   Add automatic thumbnail generation (grab a frame + overlay the hook text).
*   Add a lightweight web UI or CLI to review/approve clips before posting.
*   Wire up an upload step (YouTube/TikTok APIs) once you're happy with quality.