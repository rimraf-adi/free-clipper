import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from .logger import log_step, log_success, log_warning

def cut_clip(source_video: str, start: float, end: float, out_path: str) -> str:
    """Trims source video/audio accurately between start and end timestamps."""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    duration = max(1.0, end - start)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", source_video,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-avoid_negative_ts", "make_zero",
        out_path
    ]
    
    log_step("CutClips", f"Trimming clip [\033[1m{start:.1f}s - {end:.1f}s\033[0m] -> \033[36m{out_path}\033[0m")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        log_warning("CutClips", f"Exact seek failed for {out_path}. Falling back to copy stream seek...")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", source_video,
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            out_path
        ]
        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    return out_path

def cut_all(source_video: str, clips: List[Dict[str, Any]], out_dir: str = "clips") -> List[str]:
    """Cuts all candidate clip segments."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, clip in enumerate(clips):
        out_path = os.path.join(out_dir, f"clip_{i+1:02d}.mp4")
        cut_clip(source_video, clip["start"], clip["end"], out_path)
        paths.append(out_path)
    log_success("CutClips", f"Cut \033[1m{len(paths)}\033[0m clips into '\033[1m{out_dir}\033[0m'")
    return paths
