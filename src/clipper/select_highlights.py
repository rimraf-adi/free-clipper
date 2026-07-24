import json
import re
import time
from typing import List, Dict, Any, Optional
from .groq_client import GroqModelPool
from .logger import log_info, log_success, log_warning, log_step

FALLBACK_SYSTEM_PROMPT_TEMPLATE = """You select engaging clips strictly between {min_duration} seconds and {max_duration} seconds long from a podcast transcript that will perform well as standalone videos on social media.

Selection Guidelines:
- Target duration: Must be between {min_duration} and {max_duration} seconds long.
- Standalone clarity: The clip must be 100% understandable without needing outside context.
- Strong hook: Must start with a compelling question, bold statement, or intriguing thought in the first 3 seconds.
- Complete narrative arc: Contains a complete thought, story, insight, or punchline.
- Natural boundary: Do NOT cut mid-sentence.

Input transcript is formatted as timestamped lines: [start_sec - end_sec] text

Return ONLY a JSON array of candidate clips in this format:
[
  {{
    "start": float,
    "end": float,
    "hook": "First few words / hook line",
    "reason": "Brief explanation of why this moment is clip-worthy",
    "score": int
  }}
]
No conversational text, markdown formatting blocks, or extra comments outside the JSON array."""

def clean_json_response(raw_text: str) -> str:
    if not raw_text:
        return "[]"
    text = raw_text.strip()
    
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    
    # Extract JSON array if present
    match_arr = re.search(r"\[\s*.*?\s*\]", text, re.DOTALL)
    if match_arr:
        return match_arr.group(0)
        
    # Extract JSON object if present
    match_obj = re.search(r"\{\s*.*?\s*\}", text, re.DOTALL)
    if match_obj:
        try:
            parsed = json.loads(match_obj.group(0))
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        return json.dumps(val)
                return json.dumps([parsed])
        except Exception:
            pass
            
    return text

def chunk_transcript_by_time(transcript: List[Dict[str, Any]], window_sec: float = 300.0, overlap_sec: float = 30.0) -> List[List[Dict[str, Any]]]:
    """Splits transcript into ~5-minute overlapping windows to keep token counts within Groq TPM limits."""
    if not transcript:
        return []
        
    last_seg_end = transcript[-1].get("end", 0.0)
    if last_seg_end <= window_sec:
        return [transcript]
        
    chunks = []
    start_time = 0.0
    
    while start_time < last_seg_end:
        end_time = start_time + window_sec
        chunk_segs = [
            seg for seg in transcript
            if start_time <= seg.get("start", 0.0) <= end_time
        ]
        if chunk_segs:
            chunks.append(chunk_segs)
        start_time += (window_sec - overlap_sec)
        
    return chunks

def deduplicate_clips(clips: List[Dict[str, Any]], min_duration: float = 0.0, max_duration: float = 9999.0) -> List[Dict[str, Any]]:
    """Deduplicates overlapping candidate clips and filters strictly by min/max duration bounds."""
    if not clips:
        return []
        
    valid_clips = []
    for c in clips:
        dur = c.get("end", 0.0) - c.get("start", 0.0)
        if min_duration <= dur <= max_duration:
            valid_clips.append(c)
            
    sorted_clips = sorted(valid_clips, key=lambda c: c.get("start", 0.0))
    deduped = []
    
    for clip in sorted_clips:
        c_start = clip.get("start", 0.0)
        c_end = clip.get("end", 0.0)
        c_score = clip.get("score", 5)
        
        is_duplicate = False
        for existing in deduped:
            e_start = existing.get("start", 0.0)
            e_end = existing.get("end", 0.0)
            
            overlap = max(0.0, min(c_end, e_end) - max(c_start, e_start))
            if overlap > 10.0 or abs(c_start - e_start) < 5.0:
                is_duplicate = True
                if c_score > existing.get("score", 5):
                    existing.update(clip)
                break
                
        if not is_duplicate:
            deduped.append(clip)
            
    return sorted(deduped, key=lambda c: c.get("score", 5), reverse=True)

def select_highlights_from_chunk(
    groq_pool: GroqModelPool,
    chunk: List[Dict[str, Any]],
    chunk_idx: int,
    total_chunks: int,
    min_dur: int,
    max_dur: int,
    system_prompt_template: str = FALLBACK_SYSTEM_PROMPT_TEMPLATE
) -> List[Dict[str, Any]]:
    """Evaluates a single transcript chunk via Groq LLM for specific duration targets."""
    full_text_lines = []
    for seg in chunk:
        full_text_lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
        
    full_transcript = "\n".join(full_text_lines)
    try:
        system_prompt = system_prompt_template.format(min_duration=min_dur, max_duration=max_dur)
    except Exception:
        system_prompt = FALLBACK_SYSTEM_PROMPT_TEMPLATE.format(min_duration=min_dur, max_duration=max_dur)
        
    prompt = f"Transcript Window ({chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s):\n{full_transcript}\n\nSelect up to 3 candidate clips ({min_dur}-{max_dur}s duration). Return ONLY the JSON array."
    
    log_info("SelectHighlights", f"Evaluating window #{chunk_idx+1}/{total_chunks} ({chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s) for {min_dur}-{max_dur}s clips...")
    
    try:
        res = groq_pool.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1500
        )
        
        model_used = res["model"]
        raw_content = res.get("content") or ""
        cleaned = clean_json_response(raw_content)
        
        try:
            clips = json.loads(cleaned)
        except json.JSONDecodeError:
            log_warning("SelectHighlights", f"Could not parse JSON output from model {model_used}. Preview: {raw_content[:80]}...")
            return []
            
        if isinstance(clips, dict):
            clips = [clips]
        elif not isinstance(clips, list):
            clips = []

        valid_clips = []
        for clip in clips:
            if isinstance(clip, dict) and "start" in clip and "end" in clip and clip["end"] > clip["start"]:
                valid_clips.append(clip)
        return valid_clips
    except Exception as exc:
        log_warning("SelectHighlights", f"Failed evaluating window #{chunk_idx+1}: {exc}")
        return []

def select_hierarchical_highlights(
    transcript: List[Dict[str, Any]],
    categories_cfg: Dict[str, Any],
    system_prompt_template: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Selects hierarchical clip highlights categorized into short, mid, and long duration tiers."""
    if not transcript:
        log_warning("SelectHighlights", "Transcript is empty. Cannot select highlights.")
        return {}
        
    groq_pool = GroqModelPool()
    chunks = chunk_transcript_by_time(transcript, window_sec=300.0, overlap_sec=30.0)
    prompt_tpl = system_prompt_template or FALLBACK_SYSTEM_PROMPT_TEMPLATE
    
    categorized_clips: Dict[str, List[Dict[str, Any]]] = {}
    
    for cat_name, cat_spec in categories_cfg.items():
        if not cat_spec.get("enabled", True):
            continue
            
        min_dur = cat_spec.get("min_duration", 20)
        max_dur = cat_spec.get("max_duration", 60)
        target_count = cat_spec.get("count", 3)
        
        log_info("SelectHighlights", f"--- Extracting \033[1m{cat_name.upper()}\033[0m clips ({min_dur}s - {max_dur}s, target count: {target_count}) ---")
        
        cat_candidates = []
        for idx, chunk in enumerate(chunks):
            if chunk[-1]["end"] - chunk[0]["start"] < min_dur:
                continue
            chunk_clips = select_highlights_from_chunk(
                groq_pool, chunk, idx, len(chunks), min_dur, max_dur, system_prompt_template=prompt_tpl
            )
            cat_candidates.extend(chunk_clips)
            if idx < len(chunks) - 1:
                time.sleep(1.0)
                
        deduped = deduplicate_clips(cat_candidates, min_duration=min_dur, max_duration=max_dur)
        selected = deduped[:target_count]
        categorized_clips[cat_name] = selected
        log_success("SelectHighlights", f"Selected \033[1m{len(selected)}\033[0m {cat_name} clips ({min_dur}s-{max_dur}s).")
        
    return categorized_clips

def select_highlights(transcript: List[Dict[str, Any]], max_clips: int = 5) -> List[Dict[str, Any]]:
    """Legacy single-tier highlight selection fallback."""
    cfg = {
        "default": {"enabled": True, "min_duration": 30, "max_duration": 90, "count": max_clips}
    }
    res = select_hierarchical_highlights(transcript, cfg)
    return res.get("default", [])
