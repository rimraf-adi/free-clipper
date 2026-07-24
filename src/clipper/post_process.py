import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from .logger import log_step, log_success, log_warning, log_info

_HAS_LIBASS: Optional[bool] = None

def check_libass_support() -> bool:
    """Checks if installed FFmpeg binary has subtitle burn-in filter (libass) support."""
    global _HAS_LIBASS
    if _HAS_LIBASS is not None:
        return _HAS_LIBASS
        
    try:
        res = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            check=True
        )
        _HAS_LIBASS = "subtitles" in res.stdout
    except Exception:
        _HAS_LIBASS = False
        
    return _HAS_LIBASS

def add_captions(
    clip_path: str,
    srt_path: str,
    out_path: str,
    aspect_ratio: str = "16:9",
    sub_cfg: Optional[Dict[str, Any]] = None
) -> str:
    """Adds SRT subtitles to video clip via visual burn-in (if libass supported) or embedded MP4 subtitle track + SRT sidecar file."""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    
    sidecar_srt = os.path.splitext(out_path)[0] + ".srt"
    if os.path.exists(srt_path):
        shutil.copy2(srt_path, sidecar_srt)
        log_info("PostProcess", f"Saved sidecar subtitle file: \033[1m{sidecar_srt}\033[0m")

    sub_cfg = sub_cfg or {}
    font_name = sub_cfg.get("font_name", "Arial")
    font_size = sub_cfg.get("font_size", 18)
    bold_flag = "1" if sub_cfg.get("bold", True) else "0"
    primary_color = sub_cfg.get("primary_color", "&H00FFFFFF&")  # Pure White
    outline_color = sub_cfg.get("outline_color", "&H00000000&")  # Pure Black
    
    style = f"FontName={font_name},Bold={bold_flag},FontSize={font_size},PrimaryColour={primary_color},OutlineColour={outline_color},BorderStyle=1,Outline=2,Alignment=2,MarginV=30"
    
    has_libass = check_libass_support()
    
    if has_libass:
        escaped_srt = srt_path.replace(":", "\\:").replace("'", "\\'")
        
        if aspect_ratio == "9:16":
            vf = f"crop=ih*9/16:ih,subtitles='{escaped_srt}':force_style='{style}'"
            log_step("PostProcess", f"Rendering 9:16 vertical video + bold white subtitles -> \033[1m{out_path}\033[0m")
        else:
            vf = f"subtitles='{escaped_srt}':force_style='{style}'"
            log_step("PostProcess", f"Rendering 16:9 video + bold white subtitles -> \033[1m{out_path}\033[0m")
            
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-vf", vf,
            "-c:a", "copy",
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_success("PostProcess", f"Rendered clip: \033[1m{out_path}\033[0m")
            return out_path
        except subprocess.CalledProcessError:
            log_warning("PostProcess", "Burn-in failed. Falling back to embedded MP4 subtitles...")

    log_step("PostProcess", f"Embedding MP4 subtitle track into: \033[1m{out_path}\033[0m")
    
    if aspect_ratio == "9:16":
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-i", srt_path,
            "-vf", "crop=ih*9/16:ih",
            "-c:a", "copy",
            "-c:s", "mov_text",
            out_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-i", srt_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            out_path
        ]
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success("PostProcess", f"Rendered clip with embedded subtitles: \033[1m{out_path}\033[0m")
    except subprocess.CalledProcessError:
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-c:v", "copy",
            "-c:a", "copy",
            out_path
        ]
        log_warning("PostProcess", "Saving clip copy without subtitle track.")
        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    return out_path

to_vertical_with_captions = add_captions
