import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
from .logger import log_step, log_success, log_warning, log_info

def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """Parses relative SRT file into timestamped caption blocks."""
    if not os.path.exists(srt_path):
        return []
        
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    blocks = content.split("\n\n")
    captions = []
    
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) >= 3:
            times = lines[1].split("-->")
            if len(times) == 2:
                start = parse_srt_timestamp(times[0].strip())
                end = parse_srt_timestamp(times[1].strip())
                text = " ".join(lines[2:]).strip()
                if text:
                    captions.append({"start": start, "end": end, "text": text})
                    
    return captions

def parse_srt_timestamp(ts: str) -> float:
    """Converts HH:MM:SS,mmm timestamp to float seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0

def burn_in_subtitles_via_png_overlay(
    clip_path: str,
    srt_path: str,
    out_path: str,
    aspect_ratio: str = "16:9"
) -> bool:
    """Renders bold white visual subtitles permanently burned onto video frames using Pillow + FFmpeg overlay filter."""
    captions = parse_srt_file(srt_path)
    if not captions:
        return False

    # Get video dimensions using ffprobe
    v_width, v_height = 1920, 1080
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            clip_path
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        dims = res.stdout.strip().split("x")
        if len(dims) == 2:
            v_width, v_height = int(dims[0]), int(dims[1])
    except Exception:
        pass

    if aspect_ratio == "9:16":
        canvas_w = int(v_height * 9 / 16)
        canvas_h = v_height
    else:
        canvas_w = v_width
        canvas_h = v_height

    font_size = int(canvas_h * 0.045)  # Responsive font size
    font = None
    for f_path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]:
        if os.path.exists(f_path):
            try:
                font = ImageFont.truetype(f_path, font_size)
                break
            except Exception:
                pass
    if not font:
        font = ImageFont.load_default()

    temp_dir = tempfile.mkdtemp(prefix="captions_png_")
    filter_inputs = []
    overlay_filters = []
    
    curr_v = "[v_base]" if aspect_ratio == "9:16" else "[0:v]"
    base_filter = f"[0:v]crop=ih*9/16:ih[v_base]" if aspect_ratio == "9:16" else None

    try:
        for idx, cap in enumerate(captions):
            png_path = os.path.join(temp_dir, f"cap_{idx:03d}.png")
            img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            text = cap["text"]
            stroke_w = max(3, int(font_size * 0.08))
            y_pos = int(canvas_h * 0.82)
            
            draw.text(
                (canvas_w // 2, y_pos),
                text,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 255),
                anchor="mm"
            )
            img.save(png_path)
            
            filter_inputs.extend(["-i", png_path])
            
            in_label = curr_v
            out_label = f"v_{idx+1}"
            s_t = cap["start"]
            e_t = cap["end"]
            
            overlay_filters.append(
                f"{in_label}[{idx+1}:v]overlay=x=0:y=0:enable='between(t,{s_t:.3f},{e_t:.3f})'[{out_label}]"
            )
            curr_v = f"[{out_label}]"

        all_filters = []
        if base_filter:
            all_filters.append(base_filter)
        all_filters.extend(overlay_filters)
        
        filter_complex = ";".join(all_filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            *filter_inputs,
            "-filter_complex", filter_complex,
            "-map", curr_v,
            "-map", "0:a?",
            "-c:a", "copy",
            out_path
        ]
        
        log_step("PostProcess", f"Rendering permanent bold white visual subtitles -> \033[1m{out_path}\033[0m")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success("PostProcess", f"Rendered clip with visual burned-in subtitles: \033[1m{out_path}\033[0m")
        return True
    except Exception as exc:
        log_warning("PostProcess", f"PNG overlay burn-in failed: {exc}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def add_captions(
    clip_path: str,
    srt_path: str,
    out_path: str,
    aspect_ratio: str = "16:9",
    sub_cfg: Optional[Dict[str, Any]] = None
) -> str:
    """Adds SRT subtitles to video clip via PNG visual overlay burn-in."""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    
    sidecar_srt = os.path.splitext(out_path)[0] + ".srt"
    if os.path.exists(srt_path) and os.path.abspath(srt_path) != os.path.abspath(sidecar_srt):
        shutil.copy2(srt_path, sidecar_srt)
        log_info("PostProcess", f"Saved sidecar subtitle file: \033[1m{sidecar_srt}\033[0m")

    # Primary Method: Visual PNG Overlay Burn-In
    success = burn_in_subtitles_via_png_overlay(clip_path, srt_path, out_path, aspect_ratio=aspect_ratio)
    if success:
        return out_path

    # Fallback Method: Embedded Soft Subtitles
    log_warning("PostProcess", "Visual burn-in fallback to embedded soft subtitle track...")
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
    except Exception:
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-c:v", "copy",
            "-c:a", "copy",
            out_path
        ]
        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    return out_path

to_vertical_with_captions = add_captions
