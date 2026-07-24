"""
Podcast Clipper Package
Automated podcast and video clipping pipeline powered by Groq API and FFmpeg.
"""

from .pipeline import run_pipeline
from .config import load_config

__all__ = ["run_pipeline", "load_config"]
