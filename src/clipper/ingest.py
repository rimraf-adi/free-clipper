import os
import csv
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, cast
import yt_dlp
from .logger import log_info, log_success, log_step, log_warning

KNOWN_MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".flac"}

def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")

def sanitize_title(raw_title: str) -> str:
    """Sanitizes raw video title or path into a clean, space-stripped directory name."""
    if not raw_title:
        return "Untitled_Media"
        
    base_name = os.path.basename(raw_title).strip()
    ext = os.path.splitext(base_name)[1].lower()
    if ext in KNOWN_MEDIA_EXTENSIONS:
        base_name = os.path.splitext(base_name)[0]
        
    clean = re.sub(r"[^\w\s-]", "", base_name)
    clean = re.sub(r"[\s-]+", "_", clean).strip("_")
    return clean or "Untitled_Media"

def fetch_youtube_title(url: str) -> str:
    """Extracts YouTube video title using yt-dlp metadata API."""
    try:
        ydl_opts: Dict[str, Any] = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and isinstance(info, dict):
                title = info.get("title")
                if isinstance(title, str) and title:
                    return title
            return "YouTube_Video"
    except Exception as exc:
        log_warning("Ingest", f"Could not fetch YouTube title metadata: {exc}")
        return "YouTube_Video"

def parse_input_sources(input_source_str: str) -> List[str]:
    """Parses URLs/file paths from a CSV file path or comma-separated string."""
    if not input_source_str:
        return []
        
    s_clean = input_source_str.strip()
    
    if s_clean.lower().endswith(".csv") and os.path.isfile(s_clean):
        log_info("Ingest", f"Loading YouTube links / sources from CSV file: \033[1m{s_clean}\033[0m")
        sources = []
        with open(s_clean, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    val = cell.strip()
                    if val and not val.startswith("#"):
                        sources.append(val)
        return sources
        
    sources = [s.strip() for s in s_clean.split(",") if s.strip()]
    return sources

def download_media_from_url(url: str, out_dir: str = "work", index: int = 0) -> Dict[str, str]:
    """Downloads full video + audio from a YouTube/web URL using yt-dlp."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_title = fetch_youtube_title(url)
    sanitized = sanitize_title(raw_title)
    
    out_prefix = os.path.join(out_dir, f"yt_download_{index:02d}")
    
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.startswith(f"yt_download_{index:02d}.") and not f.endswith(".part"):
                existing_path = os.path.join(out_dir, f)
                if os.path.getsize(existing_path) > 100 * 1024:
                    log_success("Ingest", f"Found cached video: \033[1m{existing_path}\033[0m (Title: '{sanitized}')")
                    return {"path": existing_path, "title": raw_title, "sanitized_title": sanitized}
    
    out_template = f"{out_prefix}.%(ext)s"
    ydl_opts: Dict[str, Any] = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
    }
    
    log_step("Ingest", f"Downloading media #{index+1} ('\033[1m{raw_title}\033[0m') from YouTube URL: \033[36m{url}\033[0m")
    with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
        ydl.download([url])
    
    downloaded_video = f"{out_prefix}.mp4"
    if not os.path.exists(downloaded_video):
        for f in os.listdir(out_dir):
            if f.startswith(f"yt_download_{index:02d}"):
                downloaded_video = os.path.join(out_dir, f)
                break
                
    log_success("Ingest", f"Downloaded video to \033[1m{downloaded_video}\033[0m")
    return {"path": downloaded_video, "title": raw_title, "sanitized_title": sanitized}

def normalize_audio(input_source: str, out_dir: str = "work", index: int = 0) -> Dict[str, Any]:
    """Extract mono 16kHz WAV audio for Whisper while retaining full video source for clip rendering."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"audio_{index:02d}.wav")
    
    if is_url(input_source):
        media_info = download_media_from_url(input_source, out_dir, index=index)
        source_media = media_info["path"]
        raw_title = media_info["title"]
        sanitized_title = media_info["sanitized_title"]
    else:
        input_file = input_source
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        source_media = input_file
        raw_title = os.path.basename(input_file)
        sanitized_title = sanitize_title(raw_title)
            
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10 * 1024:
        log_success("Ingest", f"Found cached 16kHz mono WAV: \033[1m{out_path}\033[0m (skipping FFmpeg extraction)")
        return {
            "index": index,
            "original_source": input_source,
            "audio_wav": out_path,
            "source_media": source_media,
            "title": raw_title,
            "sanitized_title": sanitized_title,
        }
        
    log_info("Ingest", f"Extracting 16kHz mono WAV for Whisper: \033[1m{out_path}\033[0m")
    cmd = [
        "ffmpeg", "-y", "-i", source_media,
        "-ac", "1", "-ar", "16000", out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return {
        "index": index,
        "original_source": input_source,
        "audio_wav": out_path,
        "source_media": source_media,
        "title": raw_title,
        "sanitized_title": sanitized_title,
    }

def normalize_audio_sources(input_sources_str: str, out_dir: str = "work") -> List[Dict[str, Any]]:
    """Processes single/comma-separated strings or CSV files containing YouTube URLs/file paths."""
    sources = parse_input_sources(input_sources_str)
    if not sources:
        raise ValueError(f"No valid input file paths or YouTube URLs found from input: '{input_sources_str}'")
        
    results = []
    log_info("Ingest", f"Found \033[1m{len(sources)}\033[0m source input(s) to process.")
    for idx, src in enumerate(sources):
        item = normalize_audio(src, out_dir=out_dir, index=idx)
        results.append(item)
        
    return results
