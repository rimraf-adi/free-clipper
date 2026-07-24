import os
import yaml
from typing import Dict, Any

DEFAULT_SYSTEM_PROMPT = """You select engaging clips strictly between {min_duration} seconds and {max_duration} seconds long from a podcast transcript that will perform well as standalone videos on social media.

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

DEFAULT_CONFIG = {
    "output_dir": "output",
    "aspect_ratio": "16:9",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "clip_categories": {
        "short": {"enabled": True, "min_duration": 20, "max_duration": 60, "count": 3},
        "mid": {"enabled": True, "min_duration": 60, "max_duration": 180, "count": 2},
        "long": {"enabled": True, "min_duration": 180, "max_duration": 600, "count": 1},
    },
    "subtitles": {
        "font_name": "Arial",
        "font_size": 18,
        "bold": True,
        "primary_color": "&H00FFFFFF&",
        "outline_color": "&H00000000&",
        "border_style": 1,
        "outline": 2,
        "margin_v": 30,
    }
}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from config.yaml with fallback to defaults."""
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
            
        merged = dict(DEFAULT_CONFIG)
        merged.update({k: v for k, v in user_cfg.items() if v is not None})
        return merged
    except Exception as exc:
        print(f"[Config] Error loading {config_path}: {exc}. Using default config.")
        return DEFAULT_CONFIG
