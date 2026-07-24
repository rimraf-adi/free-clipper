import os
import sys
import argparse

# Add src/ directory to python import path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from clipper.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Podcast Clipper Agent Pipeline with Hierarchical Clip Extraction & Config Support"
    )
    parser.add_argument(
        "input",
        help="Path to input audio/video file, comma-separated URLs, or CSV file"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=None,
        help="Override clip count per category"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override base output directory"
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=["16:9", "9:16"],
        default=None,
        help="Override video aspect ratio (16:9 or 9:16)"
    )

    args = parser.parse_args()
    run_pipeline(
        args.input,
        max_clips=args.clips,
        output_dir=args.output_dir,
        aspect_ratio=args.aspect_ratio,
        config_file=args.config
    )

if __name__ == "__main__":
    main()
