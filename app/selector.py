"""Clip selection — Groq LLM 2-pass primary, local heuristic fallback.

PRIMARY: When Groq API keys are configured (LLM_API_KEY_1..10 in .env), uses
the 2-pass Groq LLM extraction pipeline for intelligent clip selection:
  Pass 1 — Discovery: scans transcript windows, LLM picks candidate clips.
  Pass 2 — Validation: each candidate is verified for event completeness.

FALLBACK: When no API keys are available, uses the original transparent local
heuristic: sentence-boundary-aligned windows scored by word density,
completeness, hook signals, and length fit.

An OPTIONAL, OFF-by-default scaffold can score/title windows with a *local*
Ollama model if one is detected on this machine. It still makes zero external
API calls — Ollama runs on localhost. Enable it by setting USE_OLLAMA=1.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import List

logger = logging.getLogger(__name__)

# Candidate window length bounds (seconds) and the "ideal" length we score toward.
# A clip must run at least MIN, then keeps going until the sentence finishes — it is
# never cut mid-thought. MAX is a hard safety cap (≈ for runaway monologues).
MIN_CLIP_LEN = 30.0
MAX_CLIP_LEN = 90.0
IDEAL_CLIP_LEN = 40.0
_SENT_END = (".", "!", "?", "।", "…")  # incl. Urdu/Hindi danda

# A clip can be a grammatically complete sentence and still end mid-THOUGHT —
# "and that's when I realized..." is a full stop but leaves the story hanging.
# These two signals (a real pause, or the next line sharing almost no
# vocabulary with this one) are a local, no-model stand-in for "the topic
# actually moved on" — used to prefer ending clips on a genuine story
# boundary instead of just the first sentence-end past the target length.
TOPIC_PAUSE_GAP = 0.6       # seconds of silence that reads as a natural beat
TOPIC_OVERLAP_MAX = 0.25    # word-overlap with the next line below this -> topic shifted
TOPIC_SLACK_FRAC = 0.35     # how much further past target_len to keep looking

# Words that often mark hooks / strong or curiosity-driving statements.
STRONG_WORDS = {
    "how", "why", "what", "when", "who", "where", "best", "worst", "never",
    "always", "secret", "mistake", "biggest", "important", "actually", "truth",
    "realize", "realise", "amazing", "incredible", "stop", "avoid", "must",
    "everyone", "nobody", "money", "free", "new", "first", "tip", "tips",
}

_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def _groq_keys_available() -> bool:
    """Check if any Groq API keys are configured in the environment."""
    for i in range(1, 16):
        if os.getenv(f"LLM_API_KEY_{i}"):
            return True
        if os.getenv(f"GROQ_API_KEY_{i}"):
            return True
    if os.getenv("GROQ_API_KEY"):
        return True
    return False


def _groq_score_candidates(candidates: List[dict], num_clips: int) -> List[dict]:
    """Use Groq LLM to score, validate, and title heuristic-selected candidate windows.

    The local heuristic already found structurally sound windows (complete sentences,
    topic boundaries). Now the LLM judges each candidate's viral potential and gives
    it a proper title + reason, using the 10-key intelligent router for throughput.
    """
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from clipper.groq_client import GroqModelPool

    pool = GroqModelPool()

    SCORE_PROMPT = (
        "You are scoring short-video clip candidates for social media virality. "
        "Given a transcript snippet, judge how engaging it would be as a standalone "
        "clip on TikTok/YouTube Shorts/Instagram Reels.\n\n"
        "Evaluate: hook strength (does it grab attention in 3 seconds?), "
        "completeness (is it a full story/thought?), emotional impact, and "
        "standalone clarity (understandable without context).\n\n"
        "Reply with ONLY compact JSON:\n"
        '{"score": <1-10 integer>, "title": "<<=8 word catchy hook title>", '
        '"reason": "<1 sentence explaining why this clip works or doesn\'t>", '
        '"is_complete": <true if the clip is a complete event/story, false if it cuts off>}'
    )

    # Pre-rank with heuristic score, send top 3× candidates to LLM
    pre = sorted(candidates, key=lambda c: c["score"], reverse=True)[:num_clips * 3]

    scored: List[dict] = []
    for cand in pre:
        text = cand.get("text", "")[:1500]
        duration = cand["end"] - cand["start"]
        prompt = (
            f"Clip candidate ({duration:.0f}s, from {cand['start']:.1f}s to {cand['end']:.1f}s):\n\n"
            f"{text}"
        )

        try:
            res = pool.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SCORE_PROMPT,
                temperature=0.1,
                max_tokens=200,
            )
            raw = res.get("content") or ""

            # Parse JSON response
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                # LLM didn't return valid JSON, keep candidate with heuristic score
                scored.append(cand)
                continue

            import json
            data = json.loads(match.group(0))

            llm_score = float(data.get("score", cand["score"]))
            is_complete = data.get("is_complete", True)

            # Reject incomplete clips if LLM flagged them
            if not is_complete and llm_score < 5:
                logger.info("Groq rejected incomplete clip at %.1fs-%.1fs", cand["start"], cand["end"])
                continue

            scored.append({
                "start": cand["start"],
                "end": cand["end"],
                "text": cand.get("text", ""),
                "score": llm_score,
                "title": str(data.get("title") or _derive_title(text)),
                "reason": str(data.get("reason", "")),
            })
        except Exception as exc:
            logger.warning("Groq scoring failed for candidate at %.1fs: %s", cand["start"], exc)
            scored.append(cand)  # Keep with heuristic score on LLM failure

    if not scored:
        logger.warning("All Groq scoring failed. Falling back to pure heuristic.")
        return []

    # Select best non-overlapping clips by LLM score
    scored.sort(key=lambda c: c["score"], reverse=True)
    chosen: List[dict] = []
    for cand in scored:
        if len(chosen) >= num_clips:
            break
        if any(_overlaps(cand, c) for c in chosen):
            continue
        chosen.append(cand)

    # Fill to num_clips if needed
    if len(chosen) < num_clips:
        for cand in sorted(scored, key=lambda c: c["start"]):
            if len(chosen) >= num_clips:
                break
            if any(_overlaps(cand, c) for c in chosen):
                continue
            chosen.append(cand)

    chosen.sort(key=lambda c: c["start"])
    return [
        {
            "start": c["start"],
            "end": c["end"],
            "title": c.get("title") or _derive_title(c.get("text", "")),
            "reason": c.get("reason", ""),
            "score": c.get("score", 5),
        }
        for c in chosen
    ]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def select_clips(
    transcript: dict, num_clips: int, clip_length: float | None = None
) -> List[dict]:
    """Return up to `num_clips` non-overlapping {start, end, title} windows.

    HYBRID: Local heuristics find structurally sound candidate windows (sentence
    boundaries, topic shifts, contiguous tiling), then Groq LLM scores/validates/
    titles each candidate for intelligent ranking. Falls back to pure heuristic
    when no API keys are configured.

    If ``clip_length`` is given (seconds), each window is cut to EXACTLY that
    duration. Otherwise the clip length ADAPTS to how many clips were asked for.
    """
    segments = transcript.get("segments") or []
    if not segments:
        return _fallback_even_split(transcript, num_clips, clip_length)

    duration = float(transcript.get("duration") or (segments[-1].get("end") or 0.0))

    if clip_length and clip_length > 0:
        candidates = _build_candidate_windows(segments, float(clip_length),
                                              exact_len=float(clip_length))
    else:
        target = (duration / num_clips) if (num_clips and duration) else IDEAL_CLIP_LEN
        target = max(10.0, min(MAX_CLIP_LEN, target * 0.85))
        candidates = _build_candidate_windows(segments, target)

    if not candidates:
        return _fallback_even_split(transcript, num_clips, clip_length)

    # HYBRID: Local heuristics → Groq LLM scoring
    if _groq_keys_available():
        logger.info("Groq API keys detected — using HYBRID (local heuristic + LLM scoring).")
        try:
            groq_result = _groq_score_candidates(candidates, num_clips)
            if groq_result:
                logger.info("Hybrid selection: %d clips scored and validated by LLM.", len(groq_result))
                return groq_result
        except Exception as exc:
            logger.warning("Groq hybrid scoring failed: %s. Using pure heuristic.", exc)

    # Optional local Ollama scoring (off unless explicitly enabled and available).
    if os.environ.get("USE_OLLAMA") == "1" and _ollama_available():
        try:
            return _select_with_ollama(candidates, num_clips)
        except Exception as exc:  # noqa: BLE001 - always degrade to heuristic
            logger.warning("Ollama selection failed, using heuristic: %s", exc)

    return _select_heuristic(candidates, num_clips)


# --------------------------------------------------------------------------- #
# Heuristic selection
# --------------------------------------------------------------------------- #
def _build_candidate_windows(
    segments: List[dict],
    target_len: float = IDEAL_CLIP_LEN,
    max_len: float | None = None,
    exact_len: float | None = None,
) -> List[dict]:
    """Tile the transcript into CONTIGUOUS, non-overlapping windows ~``target_len``.

    Walking start→end and jumping to each window's end (rather than one window per
    segment) packs the video tightly, so the timeline yields about
    ``duration / target_len`` windows — i.e. asking for many clips actually gives
    many. Each window still ends on a sentence boundary when possible (capped at
    ~1.2× target so the count stays close to what was requested).

    ``max_len`` overrides the hard upper cap — passed when the user picks an
    explicit clip length so a long custom length isn't clamped to ``MAX_CLIP_LEN``.

    ``exact_len`` (when set) makes every window EXACTLY that many seconds long,
    starting at a segment boundary — used when the user picks a fixed clip length
    so a "30s" clip is 30s, not ~38s snapping to the next sentence end.
    """
    if exact_len is not None and exact_len > 0:
        return _build_exact_windows(segments, exact_len)

    candidates: List[dict] = []
    n = len(segments)
    min_len = max(6.0, min(MIN_CLIP_LEN, target_len * 0.6))
    # Must comfortably exceed target_len * (1 + TOPIC_SLACK_FRAC): that's how far
    # the topic-boundary search below is allowed to look past the first
    # sentence-end, and a tighter cap here would cut it off before it gets the
    # chance (silently making the boundary search a no-op).
    if max_len is None:
        max_len = min(MAX_CLIP_LEN, max(target_len * (1.2 + TOPIC_SLACK_FRAC), min_len + 6))
    else:
        max_len = max(max_len, min_len + 6)

    i = 0
    while i < n:
        start = float(segments[i]["start"])
        end = start
        text_parts: List[str] = []
        j = i
        fallback_j: int | None = None  # first sentence-end past target_len, just in case
        while j < n:
            seg = segments[j]
            end = float(seg["end"])
            seg_text = (seg["text"] or "").strip()
            text_parts.append(seg_text)
            j += 1
            length = end - start
            if length >= target_len and seg_text.endswith(_SENT_END):
                if fallback_j is None:
                    fallback_j = j
                if _is_topic_boundary(segments, j, text_parts):
                    break  # a real story boundary, not just a complete sentence
                if length >= target_len * (1 + TOPIC_SLACK_FRAC):
                    # Kept looking for a cleaner boundary long enough — take the
                    # first acceptable sentence-end rather than run further.
                    j, end = fallback_j, float(segments[fallback_j - 1]["end"])
                    text_parts = text_parts[: fallback_j - i]
                    break
            if length >= max_len:
                if fallback_j is not None:
                    j, end = fallback_j, float(segments[fallback_j - 1]["end"])
                    text_parts = text_parts[: fallback_j - i]
                break

        i = j  # next window starts right after this one (contiguous, non-overlapping)

        length = end - start
        if length < min_len:   # the inner loop already caps the upper end near max_len
            continue

        text = " ".join(p for p in text_parts if p).strip()
        if not text:
            continue

        candidates.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "score": _score_window(text, length),
            }
        )

    return candidates


def _is_topic_boundary(segments: List[dict], j: int, window_text_parts: List[str]) -> bool:
    """Local heuristic: is the cut at ``segments[j]`` a genuine story boundary?

    True at the end of the transcript, across a real pause, or when the next
    line's vocabulary is mostly NEW relative to everything said in the window
    so far (the topic likely moved on) — as opposed to a sentence that's
    merely grammatically complete but still mid-thought (e.g. "...and that's
    when I realized"). Comparing against the WHOLE window (not just the one
    adjacent line) avoids false positives from short lines that just happen
    to phrase the same idea differently.
    """
    if j >= len(segments):
        return True
    prev_end = float(segments[j - 1]["end"])
    nxt = segments[j]
    if float(nxt["start"]) - prev_end >= TOPIC_PAUSE_GAP:
        return True
    window_words = set(re.findall(r"\b\w+\b", " ".join(window_text_parts).lower()))
    next_words = set(re.findall(r"\b\w+\b", (nxt["text"] or "").lower()))
    if not window_words or not next_words:
        return False
    # What fraction of the NEXT line's words are already-familiar (heard in
    # this window)? Low -> mostly new vocabulary -> topic likely shifted.
    familiar = len(window_words & next_words) / max(1, len(next_words))
    return familiar < TOPIC_OVERLAP_MAX


def _build_exact_windows(segments: List[dict], exact_len: float) -> List[dict]:
    """Tile the timeline into fixed ``exact_len``-second windows.

    Each window begins at a segment boundary (a natural point in the speech) and is
    cut to EXACTLY ``exact_len`` seconds, so the rendered clip is the duration the
    user asked for. Text from every segment overlapping the window is gathered for
    scoring/titling. The final window is trimmed to the media end and dropped if it
    is only a short tail.
    """
    n = len(segments)
    total_end = float(segments[-1]["end"])
    candidates: List[dict] = []

    i = 0
    while i < n:
        start = float(segments[i]["start"])
        end = min(start + exact_len, total_end)
        length = end - start

        # Gather text from every segment that overlaps [start, end).
        text_parts: List[str] = []
        j = i
        while j < n and float(segments[j]["start"]) < end:
            text_parts.append((segments[j]["text"] or "").strip())
            j += 1

        # Next window starts at the first segment beginning at/after this window's end.
        nxt = i + 1
        while nxt < n and float(segments[nxt]["start"]) < end:
            nxt += 1
        i = nxt

        # Drop a short final tail that can't make a real clip.
        if length < max(6.0, exact_len * 0.5):
            continue

        text = " ".join(p for p in text_parts if p).strip()
        if not text:
            continue

        candidates.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "score": _score_window(text, length),
            }
        )

    return candidates


def _score_window(text: str, length: float) -> float:
    """Score a window from simple, explainable local signals (higher = better)."""
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # 1) Word density: spoken-heavy windows make better clips than near-silence.
    density = word_count / max(length, 1.0)
    density_score = min(density / 3.0, 1.0)  # ~3 words/sec saturates

    # 2) Sentence completeness: rewards windows that end on a full stop.
    completeness = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.4

    # 3) Hook signals: questions and strong/curiosity words.
    strong_hits = sum(1 for w in words if w in STRONG_WORDS)
    question_bonus = 0.3 if "?" in text else 0.0
    hook_score = min(strong_hits / 5.0, 1.0) + question_bonus

    # 4) Length fit: prefer windows close to the ideal length.
    length_fit = 1.0 - min(abs(length - IDEAL_CLIP_LEN) / IDEAL_CLIP_LEN, 1.0)

    return (
        2.0 * density_score
        + 1.5 * completeness
        + 1.5 * hook_score
        + 1.0 * length_fit
    )


def _select_heuristic(candidates: List[dict], num_clips: int) -> List[dict]:
    """Pick the best non-overlapping windows, then FILL to reach `num_clips`."""
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    chosen: List[dict] = []

    # Pass 1 — quality: highest-scoring non-overlapping windows.
    for cand in ranked:
        if len(chosen) >= num_clips:
            break
        if any(_overlaps(cand, c) for c in chosen):
            continue
        chosen.append(cand)

    # Pass 2 — fill: if we still need more, add any remaining non-overlapping
    # windows in chronological order so a long video yields the count asked for.
    if len(chosen) < num_clips:
        for cand in sorted(candidates, key=lambda c: c["start"]):
            if len(chosen) >= num_clips:
                break
            if any(_overlaps(cand, c) for c in chosen):
                continue
            chosen.append(cand)

    # Present clips in chronological order.
    chosen.sort(key=lambda c: c["start"])
    return [
        {
            "start": c["start"],
            "end": c["end"],
            "title": _derive_title(c["text"]),
        }
        for c in chosen
    ]


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _derive_title(text: str, max_words: int = 7) -> str:
    """Build a short, human-readable title from the window's leading words."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split(" ")
    title = " ".join(words[:max_words]).strip(" ,.;:-")
    if not title:
        return "Clip"
    # Title-case only if it looks like all-lower/all-upper noise.
    if title.islower() or title.isupper():
        title = title.capitalize()
    return title


def _fallback_even_split(
    transcript: dict, num_clips: int, clip_length: float | None = None
) -> List[dict]:
    """Last resort: split the duration into even windows (no segments available)."""
    duration = float(transcript.get("duration") or 0.0)
    if duration <= 0:
        return []

    if clip_length and clip_length > 0:
        # Honour the requested clip length; fit as many as asked for into the video.
        clip_len = float(clip_length)
        n = max(1, min(num_clips, int(duration // clip_len) or 1))
    else:
        n = max(1, min(num_clips, 10))
        clip_len = min(MAX_CLIP_LEN, max(MIN_CLIP_LEN, duration / n))
    clips: List[dict] = []
    cursor = 0.0
    idx = 1
    while cursor < duration and len(clips) < n:
        end = min(cursor + clip_len, duration)
        if end - cursor < 3.0:  # skip a tiny tail
            break
        clips.append(
            {"start": round(cursor, 2), "end": round(end, 2), "title": f"Clip {idx}"}
        )
        cursor = end
        idx += 1
    return clips


# --------------------------------------------------------------------------- #
# Optional LOCAL Ollama scaffold (OFF by default; localhost only)
# --------------------------------------------------------------------------- #
def _ollama_available() -> bool:
    """Return True if a local Ollama server answers on localhost."""
    try:
        import requests

        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=1.5)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _select_with_ollama(candidates: List[dict], num_clips: int) -> List[dict]:
    """Score the top candidates with a local Ollama model and title them.

    This still makes NO external API calls — Ollama runs on this machine. We ask
    the model for a 0-100 interest score and a short title per window, then pick
    the best non-overlapping ones. Any failure raises so the caller can fall
    back to the pure heuristic.
    """
    import json

    import requests

    # Pre-rank with the heuristic so we only ask the local model about the best
    # ~3x candidates (keeps it fast on modest hardware).
    pre = sorted(candidates, key=lambda c: c["score"], reverse=True)[: num_clips * 3]

    scored: List[dict] = []
    for cand in pre:
        prompt = (
            "You are scoring short-video clip candidates. Given the transcript "
            "snippet, reply with ONLY compact JSON: "
            '{"score": <0-100 integer>, "title": "<<=8 word hook title>"}.\n\n'
            f"Transcript:\n{cand['text'][:1200]}"
        )
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            continue
        data = json.loads(match.group(0))
        scored.append(
            {
                "start": cand["start"],
                "end": cand["end"],
                "score": float(data.get("score", cand["score"])),
                "title": _derive_title(str(data.get("title") or cand["text"])),
            }
        )

    if not scored:
        raise RuntimeError("Ollama returned no usable scores")

    scored.sort(key=lambda c: c["score"], reverse=True)
    chosen: List[dict] = []
    for cand in scored:
        if len(chosen) >= num_clips:
            break
        if any(_overlaps(cand, c) for c in chosen):
            continue
        chosen.append(cand)

    chosen.sort(key=lambda c: c["start"])
    return [
        {"start": c["start"], "end": c["end"], "title": c["title"]} for c in chosen
    ]
