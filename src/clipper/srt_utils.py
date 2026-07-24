import os
from pathlib import Path
from typing import List, Dict, Any
from .logger import log_success

def format_timestamp(seconds: float) -> str:
    """Formats seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_for_clip(
    transcript: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    out_srt_path: str,
    max_words_per_caption: int = 4
) -> str:
    """Extracts words for clip time range and generates relative SRT subtitle file."""
    Path(os.path.dirname(out_srt_path)).mkdir(parents=True, exist_ok=True)
    
    clip_words = []
    for seg in transcript:
        words = seg.get("words", [])
        for w in words:
            w_start = w.get("start", 0.0)
            w_end = w.get("end", 0.0)
            if clip_start <= w_start <= clip_end:
                rel_start = max(0.0, w_start - clip_start)
                rel_end = max(rel_start + 0.1, w_end - clip_start)
                clip_words.append({
                    "word": w.get("word", "").strip(),
                    "start": rel_start,
                    "end": rel_end,
                })

    srt_entries = []
    idx = 1
    
    for i in range(0, len(clip_words), max_words_per_caption):
        chunk = clip_words[i:i + max_words_per_caption]
        if not chunk:
            continue
            
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        chunk_text = " ".join(w["word"] for w in chunk)
        
        start_str = format_timestamp(chunk_start)
        end_str = format_timestamp(chunk_end)
        
        srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{chunk_text}\n")
        idx += 1
        
    srt_content = "\n".join(srt_entries)
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    log_success("SRTUtils", f"Generated subtitle file: \033[1m{out_srt_path}\033[0m ({len(srt_entries)} captions)")
    return out_srt_path

def write_clip_reason_file(clip_info: Dict[str, Any], cat_name: str, out_txt_path: str) -> str:
    """Writes a text file explaining why this clip was selected by the LLM."""
    Path(os.path.dirname(out_txt_path)).mkdir(parents=True, exist_ok=True)
    
    start_sec = clip_info.get("start", 0.0)
    end_sec = clip_info.get("end", 0.0)
    duration = max(0.0, end_sec - start_sec)
    hook = clip_info.get("hook", "N/A")
    reason = clip_info.get("reason", "Selected by LLM as an engaging highlight.")
    score = clip_info.get("score", "N/A")
    
    start_fmt = format_timestamp(start_sec)
    end_fmt = format_timestamp(end_sec)
    
    content = (
        "======================================================================\n"
        "🎬 CLIP METADATA & LLM SELECTION REASONING\n"
        "======================================================================\n\n"
        f"Category:           {cat_name.upper()} ({start_sec:.1f}s - {end_sec:.1f}s)\n"
        f"Timestamps:         {start_fmt} --> {end_fmt} (Duration: {duration:.1f}s)\n"
        f"Selection Score:    {score}/10\n"
        f"Hook / Title:       \"{hook}\"\n\n"
        "----------------------------------------------------------------------\n"
        "💡 WHY THIS CLIP WAS DECIDED:\n"
        "----------------------------------------------------------------------\n"
        f"{reason}\n"
        "======================================================================\n"
    )
    
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    log_success("SRTUtils", f"Generated clip reasoning file: \033[1m{out_txt_path}\033[0m")
    return out_txt_path

