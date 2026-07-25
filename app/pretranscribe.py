"""Background pre-transcription so Whisper runs while the user is still tweaking
settings — by the time they hit Generate the transcript is already cached.

Transcription depends only on the source video + compute device (NOT on caption
style, aspect ratio, clip count, etc.), so it's safe to start as soon as the
video is on disk. Results are cached per source-file id and reused by the
pipeline, which then skips the (usually slowest) transcribe stage entirely.

A single lock serialises all transcription: the local Whisper model is shared,
and running two transcriptions at once isn't worth the risk on a single-user
tool. Double-checking the cache inside that lock means a Generate run that
arrives while a pre-transcription is mid-flight simply waits for it and then
reuses the result — the same video is never transcribed twice.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

from . import transcriber, uploads
from .models import InvalidVideoURLError, TranscriptionError
from .paths import TRANSCRIPTS_DIR

logger = logging.getLogger("ai_video_clipper.pretranscribe")


def _try_groq_whisper(
    source_path: Path,
    key: str,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Optional[dict]:
    """Attempt Groq Whisper API transcription. Returns transcript dict or None on failure."""
    # Check if any Groq API keys are available
    has_keys = False
    for i in range(1, 16):
        if os.getenv(f"LLM_API_KEY_{i}") or os.getenv(f"GROQ_API_KEY_{i}"):
            has_keys = True
            break
    if not has_keys and not os.getenv("GROQ_API_KEY"):
        return None

    try:
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from clipper.groq_client import GroqModelPool
        from clipper.transcribe import transcribe as groq_transcribe

        import subprocess

        if progress:
            progress(0.10, "Extracting audio track for Whisper API...")

        work_dir = str(TRANSCRIPTS_DIR / f"groq_{key}")
        os.makedirs(work_dir, exist_ok=True)
        audio_path = os.path.join(work_dir, "audio.wav")

        # Extract clean 16kHz mono WAV audio if not already extracted
        if not os.path.exists(audio_path):
            cmd = [
                "ffmpeg", "-y", "-i", str(source_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if progress:
            progress(0.35, "Transcribing audio with Groq Whisper API...")

        transcript_segments = groq_transcribe(audio_path, out_dir=work_dir)

        if progress:
            progress(0.85, "Processing Whisper transcript timestamps...")

        if not transcript_segments:
            logger.warning("Groq Whisper returned no segments. Falling back to local.")
            return None

        if progress:
            progress(0.95, "Groq Whisper transcription complete. Formatting...")

        # Convert clipper transcript format to ClipForge format
        total_duration = max((s.get("end", 0.0) for s in transcript_segments), default=0.0)
        segments = []
        all_words = []
        for seg in transcript_segments:
            segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
            })
            for w in seg.get("words", []):
                all_words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })

        result = {
            "language": "en",
            "duration": total_duration,
            "segments": segments,
            "words": all_words,
        }

        # Persist to disk
        tpath = TRANSCRIPTS_DIR / f"{key}.json"
        tpath.write_text(json.dumps(result, indent=2), encoding="utf-8")

        if progress:
            progress(1.0, "Groq Whisper transcription ready.")
        logger.info("Groq Whisper transcription succeeded for %s (%d segments)", key, len(segments))
        return result

    except Exception as exc:
        logger.warning("Groq Whisper transcription failed: %s. Falling back to local.", exc)
        return None

# transcript identity key (source file stem + chosen language) -> transcript dict.
# Including the language means switching languages re-transcribes instead of
# reusing a transcript decoded in the wrong language.
_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
# Serialises access to the shared Whisper model (and dedupes same-source work).
_TRANSCRIBE_LOCK = threading.Lock()


def _key(source_id: str, language: Optional[str]) -> str:
    """Cache / transcript-file key for a source at a given language ('auto' = detect)."""
    lang = (language or "auto").strip().lower() or "auto"
    return f"{source_id}__{lang}"


def cached(source_id: str, language: Optional[str] = None) -> Optional[dict]:
    """Return a cached transcript for this source+language (memory or disk), if any."""
    key = _key(source_id, language)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
    tpath = TRANSCRIPTS_DIR / f"{key}.json"
    if tpath.exists():
        try:
            result = json.loads(tpath.read_text(encoding="utf-8"))
            with _CACHE_LOCK:
                _CACHE[key] = result
            logger.info("Reusing transcript on disk for %s", key)
            return result
        except Exception:  # noqa: BLE001 - corrupt/partial json -> re-transcribe
            pass
    return None


def get_or_transcribe(
    source_path: Path,
    source_id: str,
    device: str,
    progress: Optional[Callable[[float, str], None]] = None,
    language: Optional[str] = None,
) -> dict:
    """Return a cached transcript for this source+language, or transcribe + cache it.

    Concurrency-safe: if another thread is already transcribing this source, this
    call blocks on the lock and then returns the freshly-cached result instead of
    transcribing again. Also reuses a transcript persisted on disk (survives
    restarts / in-memory cache misses). The chosen ``language`` is part of the
    cache identity, so the same video can hold a separate transcript per language.
    """
    key = _key(source_id, language)
    hit = cached(source_id, language)
    if hit is not None:
        return hit

    with _TRANSCRIBE_LOCK:
        # Re-check now that we hold the lock — a concurrent run may have finished.
        hit = cached(source_id, language)
        if hit is not None:
            return hit

        # Disk fallback: a previous run (or session) may have already saved it.
        tpath = TRANSCRIPTS_DIR / f"{key}.json"
        if tpath.exists():
            try:
                result = json.loads(tpath.read_text(encoding="utf-8"))
                with _CACHE_LOCK:
                    _CACHE[key] = result
                logger.info("Reusing transcript on disk for %s", key)
                return result
            except Exception:  # noqa: BLE001 - corrupt/partial json -> re-transcribe
                logger.warning("Ignoring unreadable transcript %s", tpath, exc_info=True)

        # PRIMARY: Try Groq Whisper API (cloud, fast) when API keys are available
        groq_result = _try_groq_whisper(source_path, key, progress=progress)
        if groq_result is not None:
            with _CACHE_LOCK:
                _CACHE[key] = groq_result
            return groq_result

        # FALLBACK: Local faster-whisper
        logger.info("Using local faster-whisper for transcription (device=%s)", device)
        result = transcriber.transcribe_video(
            source_path, key, progress=progress, device=device, language=language
        )
        with _CACHE_LOCK:
            _CACHE[key] = result
        return result


# --------------------------------------------------------------------------- #
# Background job (so the UI can show "transcribing in the background" progress)
# --------------------------------------------------------------------------- #
class TranscriptJob:
    """A single background pre-transcription with thread-safe progress state."""

    def __init__(self, source_id: str, device: str, language: Optional[str] = None) -> None:
        self.id = uuid.uuid4().hex
        self.source_id = source_id
        self.device = device
        self.language = language
        self.status = "running"  # running | done | error
        self.progress = 0.0
        self.message = "Preparing transcription..."
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._rev = 0

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)
            self._rev += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "progress": round(self.progress, 4),
                "message": self.message,
                "error": self.error,
                "rev": self._rev,
            }


_JOBS: Dict[str, TranscriptJob] = {}
_JOBS_LOCK = threading.Lock()


def get(job_id: str) -> Optional[TranscriptJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def find_running(source_id: str) -> Optional[TranscriptJob]:
    """Return a still-running background transcription for this source, if any.

    Lets the Generate pipeline mirror an in-flight pre-transcription's live
    progress (instead of a frozen bar) while it waits on the shared model lock.
    """
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job.source_id == source_id and job.status == "running":
                return job
    return None


def start(source_id: str, device: str, language: Optional[str] = None) -> TranscriptJob:
    """Begin pre-transcribing `source_id` on a daemon thread."""
    job = TranscriptJob(source_id, device, language)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    logger.info(
        "[%s] pretranscribe started: source=%s device=%s language=%s",
        job.id, source_id, device, language or "auto",
    )
    return job


def _run(job: TranscriptJob) -> None:
    if cached(job.source_id, job.language) is not None:
        job.update(status="done", progress=1.0, message="Transcript ready.")
        return
    try:
        path = uploads.resolve_upload(job.source_id)
    except InvalidVideoURLError as exc:
        job.update(status="error", error=str(exc), message="Source file not found.")
        return

    def on_progress(frac: float, msg: str) -> None:
        job.update(progress=min(0.99, frac), message=msg)

    try:
        get_or_transcribe(
            path, job.source_id, job.device, progress=on_progress, language=job.language
        )
        job.update(status="done", progress=1.0, message="Transcript ready.")
        logger.info("[%s] pretranscribe done", job.id)
    except TranscriptionError as exc:
        logger.warning("[%s] pretranscribe failed: %s", job.id, exc)
        job.update(status="error", error=str(exc), message="Could not pre-transcribe.")
    except Exception as exc:  # noqa: BLE001 - never let the thread die silently
        logger.exception("[%s] unexpected pretranscribe failure", job.id)
        job.update(
            status="error",
            error=f"Unexpected error: {exc}",
            message="Could not pre-transcribe.",
        )
