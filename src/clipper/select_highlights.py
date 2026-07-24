import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .groq_client import GroqModelPool
from .logger import log_info, log_success, log_warning, log_step

# ──────────────────────────────────────────────────────────────────────
# PASS 1: Window-Level Discovery Prompt
# ──────────────────────────────────────────────────────────────────────
DISCOVERY_PROMPT_TEMPLATE = """You select engaging clips strictly between {min_duration} seconds and {max_duration} seconds long from a podcast transcript that will perform well as standalone videos on social media.

Selection Guidelines:
- Mandatory Complete Event: The clip MUST be a 100% complete story, discussion, or event. It MUST start at the beginning of a complete sentence and end at the full conclusion of the thought or story. NEVER select incomplete events or mid-sentence cutoffs.
- Target duration: Must be between {min_duration} and {max_duration} seconds long.
- Standalone clarity: The clip must be 100% understandable without needing outside context.
- Strong hook: Must start with a compelling question, bold statement, or intriguing thought in the first 3 seconds.
- Complete narrative arc: Contains a complete thought, story, insight, or punchline.

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

# ──────────────────────────────────────────────────────────────────────
# PASS 2: Clip Validation & Boundary Refinement Prompt
# ──────────────────────────────────────────────────────────────────────
VALIDATION_PROMPT_TEMPLATE = """You are a clip quality validator. Given a transcript excerpt that was selected as a social media clip candidate, your job is to:

1. VERIFY the clip is a COMPLETE, self-contained event (story, insight, joke, or discussion). It must NOT start mid-sentence or end mid-thought.
2. REFINE the start and end timestamps so the clip begins at the very first word of the opening sentence and ends at the very last word of the concluding sentence (with punctuation like . ? !).
3. RATE the clip quality on a scale of 1-10 for social media virality.

If the clip is incomplete or cuts off mid-event, you MUST fix it by expanding the boundaries to include the full event, OR reject it entirely by returning an empty array.

Input transcript excerpt with timestamps: [start_sec - end_sec] text

Return ONLY a JSON array with exactly 0 or 1 validated clip:
[
  {{
    "start": float,
    "end": float,
    "hook": "Opening hook line of the clip",
    "reason": "Why this clip works as a standalone viral moment",
    "is_complete": true,
    "score": int
  }}
]
Return an empty array [] if the clip is unfixably incomplete or low quality (score < 5).
No conversational text, markdown formatting blocks, or extra comments outside the JSON array."""


def snap_clip_to_sentences(
    transcript: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
    min_dur: float = 15.0,
    max_dur: float = 600.0
) -> Dict[str, float]:
    """Snaps candidate timestamps to complete sentence start and end boundaries in the transcript."""
    if not transcript:
        return {"start": start_sec, "end": end_sec}
        
    start_idx = 0
    for i, seg in enumerate(transcript):
        if seg.get("start", 0.0) <= start_sec <= seg.get("end", 0.0):
            start_idx = i
            break
        elif seg.get("start", 0.0) > start_sec:
            start_idx = max(0, i - 1)
            break
            
    end_idx = start_idx
    for i in range(start_idx, len(transcript)):
        if transcript[i].get("end", 0.0) >= end_sec:
            end_idx = i
            break
            
    # Expand to complete sentence punctuation (. ? !)
    for i in range(end_idx, min(len(transcript), end_idx + 5)):
        txt = transcript[i].get("text", "").strip()
        if txt and txt[-1] in {".", "?", "!"}:
            end_idx = i
            break

    snapped_start = round(float(transcript[start_idx].get("start", start_sec)), 2)
    snapped_end = round(float(transcript[end_idx].get("end", end_sec)), 2)
    
    if snapped_end - snapped_start < min_dur and end_idx + 1 < len(transcript):
        snapped_end = round(float(transcript[end_idx + 1].get("end", snapped_end)), 2)
        
    return {"start": snapped_start, "end": snapped_end}

def clean_json_response(raw_text: str) -> str:
    if not raw_text:
        return "[]"
    text = raw_text.strip()
    
    # Strip <think>...</think> tags if present (e.g., Qwen reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
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

def extract_transcript_excerpt(
    transcript: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
    context_before: float = 10.0,
    context_after: float = 10.0
) -> List[Dict[str, Any]]:
    """Extracts transcript segments for a clip region with optional context padding for validation."""
    padded_start = max(0.0, start_sec - context_before)
    padded_end = end_sec + context_after
    return [
        seg for seg in transcript
        if padded_start <= seg.get("start", 0.0) <= padded_end
    ]

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

# ──────────────────────────────────────────────────────────────────────
# PASS 1: Discovery — Find candidate clips from each window
# ──────────────────────────────────────────────────────────────────────
def select_highlights_from_chunk(
    groq_pool: GroqModelPool,
    chunk: List[Dict[str, Any]],
    chunk_idx: int,
    total_chunks: int,
    min_dur: int,
    max_dur: int,
    system_prompt_template: str = DISCOVERY_PROMPT_TEMPLATE,
    window_cache_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """PASS 1: Evaluates a single transcript chunk via Groq LLM with per-window JSON caching."""
    window_key = f"window_{chunk_idx:02d}_{chunk[0]['start']:.1f}_{chunk[-1]['end']:.1f}_{min_dur}_{max_dur}"
    
    # Check per-window cache if available
    if window_cache_file and os.path.exists(window_cache_file):
        try:
            with open(window_cache_file, "r", encoding="utf-8") as f:
                win_cache = json.load(f)
            if isinstance(win_cache, dict) and window_key in win_cache:
                log_info("SelectHighlights", f"Loaded cached LLM window #{chunk_idx+1}/{total_chunks} ({min_dur}-{max_dur}s)")
                return win_cache[window_key]
        except Exception:
            pass

    full_text_lines = []
    for seg in chunk:
        full_text_lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
        
    full_transcript = "\n".join(full_text_lines)
    try:
        system_prompt = system_prompt_template.format(min_duration=min_dur, max_duration=max_dur)
    except Exception:
        system_prompt = DISCOVERY_PROMPT_TEMPLATE.format(min_duration=min_dur, max_duration=max_dur)
        
    prompt = f"Transcript Window ({chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s):\n{full_transcript}\n\nSelect up to 3 candidate clips ({min_dur}-{max_dur}s duration). Return ONLY the JSON array."
    
    log_info("SelectHighlights", f"Evaluating window #{chunk_idx+1}/{total_chunks} ({chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s) for {min_dur}-{max_dur}s clips...")
    
    valid_clips = []
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
            clips = []
            
        if isinstance(clips, dict):
            clips = [clips]
        elif not isinstance(clips, list):
            clips = []

        for clip in clips:
            if isinstance(clip, dict) and "start" in clip and "end" in clip and clip["end"] > clip["start"]:
                snapped = snap_clip_to_sentences(chunk, clip["start"], clip["end"], min_dur, max_dur)
                clip["start"] = snapped["start"]
                clip["end"] = snapped["end"]
                valid_clips.append(clip)
    except Exception as exc:
        log_warning("SelectHighlights", f"Failed evaluating window #{chunk_idx+1}: {exc}")

    # Save to window cache file
    if window_cache_file:
        try:
            win_cache = {}
            if os.path.exists(window_cache_file):
                with open(window_cache_file, "r", encoding="utf-8") as f:
                    win_cache = json.load(f)
            win_cache[window_key] = valid_clips
            Path(os.path.dirname(window_cache_file)).mkdir(parents=True, exist_ok=True)
            with open(window_cache_file, "w", encoding="utf-8") as f:
                json.dump(win_cache, f, indent=2)
        except Exception as exc:
            log_warning("SelectHighlights", f"Could not save window cache: {exc}")

    return valid_clips

# ──────────────────────────────────────────────────────────────────────
# PASS 2: Validation — LLM verifies each clip is a complete event
# ──────────────────────────────────────────────────────────────────────
def validate_clip_completeness(
    groq_pool: GroqModelPool,
    transcript: List[Dict[str, Any]],
    clip: Dict[str, Any],
    clip_idx: int,
    total_clips: int,
    min_dur: int,
    max_dur: int,
    validation_cache_file: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """PASS 2: Uses a second LLM call to validate and refine each candidate clip for event completeness."""
    cache_key = f"validate_{clip['start']:.1f}_{clip['end']:.1f}_{min_dur}_{max_dur}"

    # Check validation cache
    if validation_cache_file and os.path.exists(validation_cache_file):
        try:
            with open(validation_cache_file, "r", encoding="utf-8") as f:
                val_cache = json.load(f)
            if isinstance(val_cache, dict) and cache_key in val_cache:
                cached = val_cache[cache_key]
                if cached is None:
                    log_info("ClipValidator", f"Clip #{clip_idx+1} rejected (cached)")
                    return None
                log_info("ClipValidator", f"Clip #{clip_idx+1} validated (cached)")
                return cached
        except Exception:
            pass

    # Extract transcript excerpt with ±10s context for the LLM to see full picture
    excerpt = extract_transcript_excerpt(transcript, clip["start"], clip["end"], context_before=10.0, context_after=10.0)
    if not excerpt:
        return clip  # Can't validate without context, pass through

    excerpt_lines = []
    for seg in excerpt:
        marker = ""
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        if seg_start >= clip["start"] and seg_end <= clip["end"]:
            marker = " ◀ SELECTED"
        excerpt_lines.append(f"[{seg_start:.1f}-{seg_end:.1f}] {seg.get('text', '')}{marker}")

    excerpt_text = "\n".join(excerpt_lines)
    
    prompt = (
        f"Candidate Clip #{clip_idx+1}/{total_clips}:\n"
        f"  Current boundaries: {clip['start']:.1f}s - {clip['end']:.1f}s ({clip['end'] - clip['start']:.1f}s duration)\n"
        f"  Hook: \"{clip.get('hook', 'N/A')}\"\n"
        f"  Original reason: \"{clip.get('reason', 'N/A')}\"\n\n"
        f"Transcript excerpt (lines marked ◀ SELECTED are currently in the clip):\n{excerpt_text}\n\n"
        f"Validate this clip is a COMPLETE event ({min_dur}-{max_dur}s). "
        f"Fix boundaries if needed or return [] to reject. Return ONLY the JSON array."
    )

    log_step("ClipValidator", f"Validating clip #{clip_idx+1}/{total_clips} ({clip['start']:.1f}s - {clip['end']:.1f}s) for event completeness...")

    validated_clip = None
    try:
        res = groq_pool.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=VALIDATION_PROMPT_TEMPLATE,
            temperature=0.1,
            max_tokens=800
        )
        
        raw_content = res.get("content") or ""
        cleaned = clean_json_response(raw_content)
        
        try:
            results = json.loads(cleaned)
        except json.JSONDecodeError:
            log_warning("ClipValidator", f"Could not parse validation response. Passing clip through unvalidated.")
            validated_clip = clip
        else:
            if isinstance(results, dict):
                results = [results]
            if isinstance(results, list) and len(results) > 0:
                r = results[0]
                if isinstance(r, dict) and "start" in r and "end" in r:
                    # Merge validated fields into clip
                    validated_clip = {**clip, **r}
                    dur = validated_clip["end"] - validated_clip["start"]
                    
                    if dur < min_dur or dur > max_dur * 1.2:
                        log_warning("ClipValidator", f"Clip #{clip_idx+1} refined to {dur:.1f}s (outside {min_dur}-{max_dur}s). Rejecting.")
                        validated_clip = None
                    else:
                        log_success("ClipValidator", f"✔ Clip #{clip_idx+1} VALIDATED as complete event ({validated_clip['start']:.1f}s - {validated_clip['end']:.1f}s, score={validated_clip.get('score', '?')})")
            else:
                log_warning("ClipValidator", f"✘ Clip #{clip_idx+1} REJECTED by LLM as incomplete event.")
                validated_clip = None
    except Exception as exc:
        log_warning("ClipValidator", f"Validation failed for clip #{clip_idx+1}: {exc}. Passing through unvalidated.")
        validated_clip = clip

    # Save to validation cache
    if validation_cache_file:
        try:
            val_cache = {}
            if os.path.exists(validation_cache_file):
                with open(validation_cache_file, "r", encoding="utf-8") as f:
                    val_cache = json.load(f)
            val_cache[cache_key] = validated_clip
            Path(os.path.dirname(validation_cache_file)).mkdir(parents=True, exist_ok=True)
            with open(validation_cache_file, "w", encoding="utf-8") as f:
                json.dump(val_cache, f, indent=2)
        except Exception as exc:
            log_warning("ClipValidator", f"Could not save validation cache: {exc}")

    return validated_clip


def select_hierarchical_highlights(
    transcript: List[Dict[str, Any]],
    categories_cfg: Dict[str, Any],
    system_prompt_template: Optional[str] = None,
    out_dir: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """Selects hierarchical clip highlights using 2-pass LLM extraction:
    
    PASS 1 (Discovery): Scans transcript windows to find candidate clips.
    PASS 2 (Validation): Each candidate is verified by a second LLM call for event completeness,
                          rejecting incomplete clips and refining boundaries.
    """
    if not transcript:
        log_warning("SelectHighlights", "Transcript is empty. Cannot select highlights.")
        return {}
        
    highlights_cache_file = os.path.join(out_dir, "llm_highlights.json") if out_dir else None
    window_cache_file = os.path.join(out_dir, "llm_window_cache.json") if out_dir else None
    validation_cache_file = os.path.join(out_dir, "llm_validation_cache.json") if out_dir else None

    # Tier 1: Overall Highlights Cache
    if highlights_cache_file and os.path.exists(highlights_cache_file) and not force_refresh:
        try:
            with open(highlights_cache_file, "r", encoding="utf-8") as f:
                cached_highlights = json.load(f)
            if isinstance(cached_highlights, dict) and cached_highlights:
                log_success("SelectHighlights", f"Loaded cached LLM highlights from \033[1m{highlights_cache_file}\033[0m")
                return cached_highlights
        except Exception as exc:
            log_warning("SelectHighlights", f"Could not read LLM highlights cache {highlights_cache_file}: {exc}")

    groq_pool = GroqModelPool()
    chunks = chunk_transcript_by_time(transcript, window_sec=300.0, overlap_sec=30.0)
    prompt_tpl = system_prompt_template or DISCOVERY_PROMPT_TEMPLATE
    
    categorized_clips: Dict[str, List[Dict[str, Any]]] = {}
    
    for cat_name, cat_spec in categories_cfg.items():
        if not cat_spec.get("enabled", True):
            continue
            
        min_dur = cat_spec.get("min_duration", 20)
        max_dur = cat_spec.get("max_duration", 60)
        target_count = cat_spec.get("count", 3)
        
        log_info("SelectHighlights", f"━━━ PASS 1: Discovery — \033[1m{cat_name.upper()}\033[0m clips ({min_dur}s - {max_dur}s, target: {target_count}) ━━━")
        
        # ── PASS 1: Discovery ──
        cat_candidates = []
        for idx, chunk in enumerate(chunks):
            if chunk[-1]["end"] - chunk[0]["start"] < min_dur:
                continue
            chunk_clips = select_highlights_from_chunk(
                groq_pool, chunk, idx, len(chunks), min_dur, max_dur,
                system_prompt_template=prompt_tpl,
                window_cache_file=window_cache_file
            )
            cat_candidates.extend(chunk_clips)
            if idx < len(chunks) - 1 and not window_cache_file:
                time.sleep(1.0)
                
        deduped = deduplicate_clips(cat_candidates, min_duration=min_dur, max_duration=max_dur)
        
        # Take more candidates than target for validation pass to filter down
        candidates_for_validation = deduped[:target_count * 2]
        
        log_info("SelectHighlights", f"━━━ PASS 2: Validation — Verifying \033[1m{len(candidates_for_validation)}\033[0m {cat_name} candidate(s) for complete events ━━━")
        
        # ── PASS 2: Validation ──
        validated = []
        for v_idx, candidate in enumerate(candidates_for_validation):
            result = validate_clip_completeness(
                groq_pool, transcript, candidate, v_idx, len(candidates_for_validation),
                min_dur, max_dur,
                validation_cache_file=validation_cache_file
            )
            if result is not None:
                validated.append(result)
                
        # Re-deduplicate after validation may have shifted boundaries
        validated = deduplicate_clips(validated, min_duration=min_dur, max_duration=max_dur)
        selected = validated[:target_count]
        categorized_clips[cat_name] = selected
        log_success("SelectHighlights", f"Selected \033[1m{len(selected)}\033[0m validated {cat_name} clips ({min_dur}s-{max_dur}s).")

    # Save overall highlights cache
    if highlights_cache_file:
        try:
            Path(os.path.dirname(highlights_cache_file)).mkdir(parents=True, exist_ok=True)
            with open(highlights_cache_file, "w", encoding="utf-8") as f:
                json.dump(categorized_clips, f, indent=2)
            log_success("SelectHighlights", f"Saved LLM highlights cache to \033[1m{highlights_cache_file}\033[0m")
        except Exception as exc:
            log_warning("SelectHighlights", f"Could not save LLM highlights cache: {exc}")

    return categorized_clips

def select_highlights(transcript: List[Dict[str, Any]], max_clips: int = 5) -> List[Dict[str, Any]]:
    """Legacy single-tier highlight selection fallback."""
    cfg = {
        "default": {"enabled": True, "min_duration": 30, "max_duration": 90, "count": max_clips}
    }
    res = select_hierarchical_highlights(transcript, cfg)
    return res.get("default", [])
