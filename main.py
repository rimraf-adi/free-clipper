import os
import sys
import argparse

# Add src/ directory to python import path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Video Clipper — CLI Pipeline + Web UI Server"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to input audio/video file, comma-separated URLs, or CSV file (CLI mode)"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch the Web UI server instead of running the CLI pipeline"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port (default: 8000)"
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
        help="Override clip count per category (CLI mode)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override base output directory (CLI mode)"
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=["16:9", "9:16"],
        default=None,
        help="Override video aspect ratio (CLI mode)"
    )

    args = parser.parse_args()

    if args.server:
        # Web UI mode — launch FastAPI server
        import uvicorn
        print(f"\n🎬 ClipForge Web UI starting at http://{args.host}:{args.port}\n")
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
    elif args.input:
        # CLI mode — run pipeline
        from clipper.pipeline import run_pipeline
        run_pipeline(
            args.input,
            max_clips=args.clips,
            output_dir=args.output_dir,
            aspect_ratio=args.aspect_ratio,
            config_file=args.config
        )
    else:
        parser.print_help()
        print("\n💡 Use --server to launch the Web UI, or pass an input file/URL for CLI mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
