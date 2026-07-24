import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import load_config
from .ingest import normalize_audio_sources
from .transcribe import transcribe
from .select_highlights import select_hierarchical_highlights
from .cut_clips import cut_clip
from .srt_utils import generate_srt_for_clip
from .post_process import add_captions
from .logger import print_header, print_stage_banner, print_summary_box, log_info, log_success, log_warning

def run_pipeline(
    input_source: str,
    max_clips: Optional[int] = None,
    output_dir: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    config_file: str = "config.yaml"
) -> List[str]:
    """Runs the podcast clipping agent pipeline with title-based directory organization and hierarchical clip extraction."""
    cfg = load_config(config_file)
    
    if output_dir:
        cfg["output_dir"] = output_dir
    if aspect_ratio:
        cfg["aspect_ratio"] = aspect_ratio
    if max_clips and max_clips > 0:
        for cat in cfg["clip_categories"].values():
            cat["count"] = max_clips

    print_header(f"Podcast Clipper Pipeline (Groq Powered) [{cfg['aspect_ratio']} Mode]")
    
    # Stage 1: Audio Ingest & Normalization
    print_stage_banner(1, "Audio Ingest & Normalization")
    items = normalize_audio_sources(input_source, out_dir="work")
    
    all_final_clip_paths = []
    
    for item in items:
        idx = item["index"]
        original = item["original_source"]
        audio_path = item["audio_wav"]
        source_media = item["source_media"]
        sanitized_title = item["sanitized_title"]
        
        log_info("Pipeline", f"\033[1mProcessing Source #{idx+1}/{len(items)}:\033[0m {sanitized_title} ({original})")
        
        title_output_dir = os.path.join(cfg["output_dir"], sanitized_title)
        Path(title_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Stage 2: Transcription
        print_stage_banner(2, f"Transcription ({sanitized_title})")
        work_sub_dir = os.path.join("work", f"source_{idx:02d}")
        transcript = transcribe(audio_path, out_dir=work_sub_dir)
        if not transcript:
            log_warning("Pipeline", f"Transcription for source #{idx+1} failed or returned empty.")
            continue
            
        # Stage 3: Hierarchical Highlight Selection (Short, Mid, Long)
        print_stage_banner(3, f"Hierarchical Highlight Selection ({sanitized_title})")
        categorized_highlights = select_hierarchical_highlights(
            transcript,
            cfg["clip_categories"],
            system_prompt_template=cfg.get("system_prompt")
        )
        
        total_clips_found = sum(len(v) for v in categorized_highlights.values())
        if total_clips_found == 0:
            log_warning("Pipeline", f"No clip highlights selected for '{sanitized_title}'.")
            continue
            
        # Stage 4 & 5: Cut Clips & Post-Process by Category
        print_stage_banner(4, f"Cutting & Post-Processing Clips ({sanitized_title})")
        
        for cat_name, clips in categorized_highlights.items():
            if not clips:
                continue
                
            cat_out_dir = os.path.join(title_output_dir, f"{cat_name}_clips")
            Path(cat_out_dir).mkdir(parents=True, exist_ok=True)
            clips_work_dir = os.path.join("clips", f"source_{idx:02d}", f"{cat_name}_clips")
            
            log_info("Pipeline", f"Processing \033[1m{len(clips)}\033[0m {cat_name} clip(s) -> \033[36m{cat_out_dir}\033[0m")
            
            for i, clip in enumerate(clips):
                raw_clip_path = os.path.join(clips_work_dir, f"clip_{i+1:02d}.mp4")
                cut_clip(source_media, clip["start"], clip["end"], raw_clip_path)
                
                srt_path = os.path.join(work_sub_dir, f"{cat_name}_clip_{i+1:02d}.srt")
                generate_srt_for_clip(transcript, clip["start"], clip["end"], srt_path)
                
                final_out = os.path.join(cat_out_dir, f"clip_{i+1:02d}.mp4")
                add_captions(
                    raw_clip_path,
                    srt_path,
                    final_out,
                    aspect_ratio=cfg["aspect_ratio"],
                    sub_cfg=cfg["subtitles"]
                )
                all_final_clip_paths.append(final_out)
                
    summary_items = [f"Total clips generated: \033[1m{len(all_final_clip_paths)}\033[0m organized in '\033[1m{cfg['output_dir']}\033[0m'"]
    for p in all_final_clip_paths:
        summary_items.append(f"  -> \033[36m{p}\033[0m")
        
    print_summary_box("Hierarchical Pipeline Completed Successfully!", summary_items)
    
    return all_final_clip_paths
