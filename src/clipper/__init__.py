"""
Podcast Clipper Package
Automated podcast and video clipping pipeline powered by Groq API and FFmpeg.
"""

from clipper.pipeline import run_pipeline
from clipper.config import load_config

__all__ = ["run_pipeline", "load_config"]
