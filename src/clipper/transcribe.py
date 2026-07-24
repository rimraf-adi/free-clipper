import json
import os
import subprocess
import wave
from pathlib import Path
from typing import List, Dict, Any, Optional
from clipper.groq_client import GroqModelPool
from clipper.logger import log_info, log_success, log_warning, log_step, log_error

MAX_GROQ_FILE_SIZE_BYTES = 20 * 1024 * 1024
CHUNK_DURATION_SEC = 600.0

def get_audio_duration(audio_path: str) -> float:
    """Returns audio duration in seconds using python wave module."""
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())

def split_audio_into_chunks(audio_path: str, chunk_dir: str, chunk_duration: float = CHUNK_DURATION_SEC) -> List[Dict[str, Any]]:
    """Splits an audio file into sub-20MB chunks using FFmpeg fast stream copy."""
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    total_duration = get_audio_duration(audio_path)
    
    chunks = []
    offset = 0.0
    idx = 0
    
    log_info("AudioSplitter", f"Total duration: \033[1m{total_duration:.1f}s\033[0m. Splitting into {chunk_duration/60:.0f}-min chunks...")
    
    while offset < total_duration:
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:03d}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(offset),
            "-i", audio_path,
            "-t", str(chunk_duration),
            "-c", "copy",
            chunk_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        chunks.append({
            "path": chunk_path,
            "offset": offset,
            "index": idx,
        })
        
        offset += chunk_duration
        idx += 1
        
    log_success("AudioSplitter", f"Created \033[1m{len(chunks)}\033[0m chunks for Groq API processing.")
    return chunks

def to_dict(obj: Any) -> Any:
    """Helper to convert Pydantic objects, dicts, or mocks into plain Python dicts."""
    if isinstance(obj, dict):
        return obj
    if "unittest.mock" in type(obj).__module__:
        res = {}
        for k in ["start", "end", "text", "word", "segments", "words"]:
            if k in obj.__dict__:
                val = obj.__dict__[k]
                res[k] = val
        return res
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    res = {}
    for k in ["start", "end", "text", "word", "segments", "words"]:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if not callable(val):
                res[k] = val
    return res

def parse_groq_response(response_data: Any, time_offset: float = 0.0) -> List[Dict[str, Any]]:
    """Converts Groq transcription response into a clean, compact, JSON-serializable list."""
    res_dict = to_dict(response_data)
    transcript = []
    
    raw_segments = res_dict.get("segments", []) or []
    raw_words = res_dict.get("words", []) or []
    
    clean_words = []
    for w in raw_words:
        w_d = to_dict(w)
        w_text = str(w_d.get("word") or w_d.get("text") or "").strip()
        if not w_text or w_text.lower() == "none":
            continue
        w_start = float(w_d.get("start", 0.0) or 0.0) + time_offset
        w_end = float(w_d.get("end", 0.0) or 0.0) + time_offset
        clean_words.append({
            "word": w_text,
            "start": round(w_start, 3),
            "end": round(w_end, 3),
        })

    if raw_segments:
        w_idx = 0
        num_words = len(clean_words)
        
        for seg in raw_segments:
            seg_d = to_dict(seg)
            seg_text = str(seg_d.get("text", "") or "").strip()
            if seg_text.lower() == "none":
                seg_text = ""
            seg_start = float(seg_d.get("start", 0.0) or 0.0) + time_offset
            seg_end = float(seg_d.get("end", 0.0) or 0.0) + time_offset
            
            seg_direct_words = seg_d.get("words", []) or []
            seg_words = []
            
            if seg_direct_words and isinstance(seg_direct_words, list):
                for dw in seg_direct_words:
                    dw_d = to_dict(dw)
                    dw_text = str(dw_d.get("word") or dw_d.get("text") or "").strip()
                    if dw_text and dw_text.lower() != "none":
                        seg_words.append({
                            "word": dw_text,
                            "start": round(float(dw_d.get("start", 0.0) or 0.0) + time_offset, 3),
                            "end": round(float(dw_d.get("end", 0.0) or 0.0) + time_offset, 3),
                        })
            else:
                while w_idx < num_words and clean_words[w_idx]["start"] < seg_start - 0.05:
                    w_idx += 1
                curr = w_idx
                while curr < num_words and clean_words[curr]["start"] <= seg_end + 0.05:
                    seg_words.append(clean_words[curr])
                    curr += 1

            if seg_text or seg_words:
                transcript.append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": seg_text,
                    "words": seg_words,
                })
                
    elif clean_words:
        full_text = str(res_dict.get("text", "") or "").strip()
        transcript.append({
            "start": clean_words[0]["start"],
            "end": clean_words[-1]["end"],
            "text": full_text if full_text.lower() != "none" else "",
            "words": clean_words,
        })
    elif res_dict.get("text"):
        full_text = str(res_dict.get("text")).strip()
        if full_text and full_text.lower() != "none":
            transcript.append({
                "start": round(time_offset, 3),
                "end": round(time_offset, 3),
                "text": full_text,
                "words": [],
            })
            
    return transcript

def transcribe_with_groq(audio_path: str, out_dir: str = "work") -> Optional[List[Dict[str, Any]]]:
    """Transcribes audio using Groq Whisper API, automatically chunking large files (>20MB) to prevent 413 errors."""
    groq_pool = GroqModelPool()
    file_size = os.path.getsize(audio_path)
    
    if file_size > MAX_GROQ_FILE_SIZE_BYTES:
        log_info("TranscribeGroq", f"File size (\033[1m{file_size / (1024*1024):.1f}MB\033[0m) > 20MB. Auto-chunking audio...")
        chunk_dir = os.path.join(out_dir, "groq_chunks")
        chunks = split_audio_into_chunks(audio_path, chunk_dir)
        
        full_transcript = []
        for c in chunks:
            log_step("TranscribeGroq", f"Transcribing chunk #{c['index']+1}/{len(chunks)}...")
            res = groq_pool.transcribe_audio(c["path"])
            if not res:
                log_warning("TranscribeGroq", f"Failed to transcribe chunk #{c['index']+1}")
                return None
            chunk_transcript = parse_groq_response(res, time_offset=c["offset"])
            full_transcript.extend(chunk_transcript)
            
            if os.path.exists(c["path"]):
                os.remove(c["path"])
                
        return full_transcript
    else:
        log_info("TranscribeGroq", f"Sending audio file (\033[1m{file_size / (1024*1024):.1f}MB\033[0m) to Groq Whisper API...")
        res = groq_pool.transcribe_audio(audio_path)
        if res:
            return parse_groq_response(res, time_offset=0.0)
        return None

def transcribe_local(audio_path: str, model_size: str = "small") -> List[Dict[str, Any]]:
    """Local transcription fallback using faster-whisper."""
    from faster_whisper import WhisperModel
    
    log_info("TranscribeLocal", f"Running local faster-whisper (model=\033[1m{model_size}\033[0m)...")
    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    transcript = []
    for seg in segments:
        text = seg.text.strip()
        words = []
        for w in (seg.words or []):
            w_text = w.word.strip()
            if w_text:
                words.append({"word": w_text, "start": round(w.start, 3), "end": round(w.end, 3)})
        if text or words:
            transcript.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": words
            })
    return transcript

def transcribe(audio_path: str, out_dir: str = "work", model_size: str = "small") -> List[Dict[str, Any]]:
    """Transcribes audio, skipping if cached transcript.json exists, trying Groq Whisper first with local fallback."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(out_dir, "transcript.json")
    
    if os.path.exists(out_json) and os.path.getsize(out_json) > 10:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                cached_transcript = json.load(f)
            if cached_transcript and isinstance(cached_transcript, list):
                log_success("Transcribe", f"Found cached transcript: \033[1m{out_json}\033[0m ({len(cached_transcript)} segments) (skipping transcription)")
                return cached_transcript
        except Exception:
            log_warning("Transcribe", f"Cached transcript {out_json} invalid. Re-transcribing...")
    
    transcript = None
    
    try:
        transcript = transcribe_with_groq(audio_path, out_dir=out_dir)
    except Exception as exc:
        log_warning("Transcribe", f"Groq Whisper API error: {exc}")
        
    if not transcript:
        log_warning("Transcribe", "Falling back to local faster-whisper model...")
        transcript = transcribe_local(audio_path, model_size=model_size)

    tmp_json = out_json + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    os.replace(tmp_json, out_json)
        
    log_success("Transcribe", f"Saved transcript (\033[1m{len(transcript)} segments\033[0m) to: \033[1m{out_json}\033[0m")
    return transcript
