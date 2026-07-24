# 🎬 Free Podcast & Video Clipper Agent

An automated, AI-powered podcast and video clipping pipeline built with **Python**, **Groq API** (round-robin model pool), **Whisper**, and **FFmpeg**.

It automatically ingests raw podcasts (local files or YouTube URLs/CSVs), transcribes audio with word-level timestamps, uses an LLM agent to select viral candidate moments, and renders high-definition widescreen (16:9) or vertical (9:16) clips with bold white subtitles.

---

## 🔥 Features

- **Hierarchical Clip Extraction (`short`, `mid`, `long`)**: Extract customized clip counts per duration tier (Short: 20-60s, Mid: 1-3 mins, Long: 3-10 mins).
- **Groq Model Pool with Round-Robin & Failover**: Automatically rotates through LLM models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `qwen/qwen3.6-27b`, `groq/compound`) with rate limit handling.
- **Transcript Window Chunking**: Avoids Groq TPM rate limits by analyzing transcripts in 5-minute windows.
- **YouTube Title Directory Naming**: Automatically organizes output clips into directories named after sanitized YouTube video titles.
- **Smart Caching**: Skips re-downloading videos, re-extracting audio, or re-transcribing if files already exist on disk.
- **Bold White Subtitles & Sidecar `.srt` Export**: Embeds MP4 subtitle tracks and exports companion `.srt` subtitle files alongside videos.
- **Config-Driven Prompt & Settings**: All prompt guidelines, aspect ratios, clip limits, and styling are managed via `config.yaml`.

---

## 📁 Repository Structure

```
clipper/
├── src/
│   └── clipper/
│       ├── __init__.py
│       ├── config.py          # Configuration loader (config.yaml)
│       ├── logger.py          # ANSI terminal logger
│       ├── groq_client.py     # Groq LLM & Whisper model pool
│       ├── ingest.py          # Media downloader & WAV extractor
│       ├── transcribe.py      # Groq Whisper API & local faster-whisper
│       ├── select_highlights.py # Hierarchical LLM clip extraction
│       ├── srt_utils.py       # SRT subtitle generator
│       ├── cut_clips.py       # Frame-accurate clip trimmer
│       ├── post_process.py    # Subtitle embedding & aspect ratio renderer
│       └── pipeline.py        # End-to-end pipeline orchestrator
├── main.py                    # CLI entry point launcher
├── config.yaml                # Editable configuration file
├── links.csv                  # Example YouTube URL list
├── pyproject.toml             # Package definition & Pyright config
├── .env.example               # Template for API key
├── README.md
└── spec.md
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`
- **[FFmpeg](https://ffmpeg.org/)** installed on system `PATH`

### 2. Environment Setup

Copy `.env.example` to `.env` and set your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Installation

Using `uv`:
```bash
uv sync
```

Alternatively, install in editable mode:
```bash
uv pip install -e .
```

---

## 🚀 CLI Commands & Usage

### Basic Usage

- **Process a Single YouTube URL**:
  ```bash
  uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
  ```

- **Process a Local Video or Audio File**:
  ```bash
  uv run python main.py "/path/to/local/podcast.mp4"
  ```

- **Process YouTube URLs from a CSV File**:
  ```bash
  uv run python main.py links.csv
  ```

- **Process Multiple Comma-Separated Links**:
  ```bash
  uv run python main.py "https://youtu.be/ID1, https://youtu.be/ID2"
  ```

---

### Command Flags & Options

| Command Flag | Description | Example |
| :--- | :--- | :--- |
| `input` | **(Required)** Path to video file, URL, comma-separated URLs, or CSV file | `uv run python main.py video.mp4` |
| `--config` | Custom YAML configuration file path *(default: `config.yaml`)* | `uv run python main.py links.csv --config custom.yaml` |
| `--aspect-ratio` | Video format ratio: `16:9` (Widescreen) or `9:16` (Vertical Shorts/Reels) | `uv run python main.py links.csv --aspect-ratio 9:16` |
| `--clips` | Override max clip count per category | `uv run python main.py links.csv --clips 3` |
| `--output-dir` | Custom destination directory for exported clips | `uv run python main.py links.csv --output-dir my_clips` |

---

## ⚙️ Configuration (`config.yaml`)

Customize extraction categories, LLM prompts, and subtitle styling in `config.yaml`:

```yaml
output_dir: "output"
aspect_ratio: "16:9"

system_prompt: |
  You select engaging clips strictly between {min_duration} seconds and {max_duration} seconds...

clip_categories:
  short:
    enabled: true
    min_duration: 20
    max_duration: 60
    count: 3
  mid:
    enabled: true
    min_duration: 60
    max_duration: 180
    count: 2
  long:
    enabled: true
    min_duration: 180
    max_duration: 600
    count: 1

subtitles:
  font_name: "Arial"
  font_size: 18
  bold: true
  primary_color: "&H00FFFFFF&"   # Bold White text
  outline_color: "&H00000000&"   # Black outline
```

---

## 📂 Output Layout

All output videos and sidecar subtitle files are organized cleanly by title:

```
output/
└── [Sanitized_YouTube_Title]/
    ├── short_clips/
    │   ├── clip_01.mp4
    │   ├── clip_01.srt
    │   └── ...
    ├── mid_clips/
    │   ├── clip_01.mp4
    │   ├── clip_01.srt
    │   └── ...
    └── long_clips/
        ├── clip_01.mp4
        ├── clip_01.srt
        └── ...
```

---

## 🧪 Testing & Verification

- **Run Unit Tests**:
  ```bash
  uv run python test_pipeline.py
  ```

- **Run Type Checker (Pyright)**:
  ```bash
  npx pyright
  ```

---

## 📄 License
MIT License
