# 🎬 Free AI Video Clipper & Podcast Studio

An automated, high-performance video clipping studio and CLI pipeline powered by **Groq API (10-key intelligent rotation pool)**, **Whisper**, **FFmpeg**, and a **React + FastAPI Web Dashboard**.

Turn any podcast, webinar, or long video into engaging, reframed, captioned short clips (9:16 Shorts/Reels/TikTok or 16:9 Widescreen) — automatically selecting complete narrative events with AI.

---

## 🌟 Key Features

### 💻 Dual Interface: Web UI & CLI
- **React Web Dashboard**: Modern UI for URL pasting, file uploads, live SSE progress rendering, timeline preview, reframe editor, and music customizer. Launch with `uv run python main.py --server`.
- **CLI Pipeline**: High-throughput terminal agent mode for batch processing local files, YouTube URLs, or CSV lists. Launch with `uv run python main.py links.csv`.

### 🧠 10-Key Intelligent Groq Router (1,000,000+ TPD Capacity)
- **2D Rotation Matrix (10 Keys × 4 Models)**: Automatically rotates API keys (`LLM_API_KEY_1`..`10`) and models (`llama-3.1-8b-instant`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`).
- **Dynamic 429 Cooldown Tracking**: Instant 0ms failover when a key or model hits a rate limit.
- **Groq-First with Local Fallback**: Uses Groq Whisper API (cloud) and falls back seamlessly to local `faster-whisper` (GPU CUDA / CPU).

### 🎯 Hybrid & 2-Pass Intelligent Event Extraction
- **Hybrid Selection**: Local heuristics tile the transcript into structurally sound candidate windows (sentence boundaries, topic shifts), then Groq LLMs score, validate, and title each candidate for social media virality.
- **2-Pass Validation**: Pass 1 discovers candidate moments; Pass 2 passes each candidate to a validator LLM with ±10s context padding to guarantee 100% complete story arcs (no awkward mid-sentence cutoffs).

### 🎨 Rich ASS Subtitle Engine & Styles
- Preset styles: **Bold White**, **Karaoke Yellow**, **Hormozi Green**, **Beast Pop**, **Raj Shamani Clean**, **Alex Bold Caps**, **One-Word Punch**, **Word Reveal**, **Bebas Clean**, and more.
- Per-word timing animation and live on-screen CSS preview matching burned-in ASS output.

### 🎭 Cinematic Effects & Background Music
- **Color Grading & Visual Effects**: Warm, Cool, Teal & Orange, Vintage, Vibrant, B&W presets with custom top/bottom gradient bands and glow bloom.
- **Background Music Library**: Drop custom audio tracks into `assets/music/` or upload via UI, with automated voice ducking and volume control.
- **Cropping & Reframing**: Crop-fill mode, 1:1 rounded-square reel canvas, and interactive manual crop reframe editor.

### ⚙️ GUI Settings & Configuration Management
- Live view and update of `config.yaml` and `.env` API keys directly from the Web UI Settings tab.

---

## 📁 Project Layout

```
clipper/
├── app/                       # FastAPI Web Backend (Jobs, Endpoints, Transcriber, Selector)
├── src/clipper/               # Core Pipeline Package (Groq Router, Pipeline, Config, Ingest)
├── web/                       # React Dashboard Frontend (Vite + React)
│   ├── src/                   # React Components, Pages, and Hooks
│   └── dist/                  # Production-built web app bundle
├── assets/                    # Assets (Fonts, Masks, Background Music Library)
├── config.yaml                # Editable configuration file
├── main.py                    # Dual Entry Point Launcher (CLI & Web Server)
├── links.csv                  # Example YouTube URL list
├── pyproject.toml             # Package dependencies & Pyright config
└── .env.example               # Template for API keys
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`
- **[FFmpeg](https://ffmpeg.org/)** installed on your system `PATH`
- **Node.js 18+** (only if building/customizing the React web dashboard)

### 2. Environment Setup

Copy `.env.example` to `.env` and set your Groq API keys:

```bash
cp .env.example .env
```

Edit `.env` (add 1 to 10 Groq keys for rotation):
```env
LLM_API_KEY_1=gsk_your_key_1
LLM_API_KEY_2=gsk_your_key_2
LLM_API_KEY_3=gsk_your_key_3
...
LLM_API_KEY_10=gsk_your_key_10
```

### 3. Installation

Install all Python dependencies using `uv`:
```bash
uv sync
```

---

## 🚀 Running the Application

### 🌐 Option A: Web UI Studio Mode

Launch the web application server:

```bash
uv run python main.py --server
```

Then open **`http://127.0.0.1:8000`** in your browser to access the Web UI Studio:
- **Paste Link or Upload Video**: Input YouTube URLs or local video files.
- **Select Aspect Ratio & Fit Mode**: 9:16 Shorts/Reels or 16:9 Widescreen; Crop-Fill or 1:1 Rounded Square.
- **Customization Studio**: Select caption style presets, cinematic color grades, background music tracks, and ducking volume.
- **Live SSE Progress Stream**: Watch stage-by-stage progress (Download $\rightarrow$ Transcribe $\rightarrow$ Analyze $\rightarrow$ Render).
- **Settings Tab**: Edit `config.yaml` parameters and `.env` API keys interactively.

---

### 💻 Option B: CLI Batch Pipeline Mode

Run batch clipping directly from the command line:

- **Single YouTube Video**:
  ```bash
  uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
  ```

- **Local Video or Audio File**:
  ```bash
  uv run python main.py "/path/to/podcast.mp4"
  ```

- **Batch List from CSV File**:
  ```bash
  uv run python main.py links.csv
  ```

- **CLI Flag Overrides**:
  ```bash
  uv run python main.py links.csv --aspect-ratio 9:16 --clips 3 --output-dir my_shorts
  ```

---

## ⚙️ Configuration (`config.yaml`)

Manage defaults in `config.yaml` (or edit live in the Web UI Settings tab):

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

## 🧪 Testing & Verification

- **Run Unit Tests**:
  ```bash
  uv run python test_pipeline.py
  ```

- **Run Type Checker (Pyright)**:
  ```bash
  npx pyright
  ```

- **Rebuild Web UI Bundle** (after modifying `web/src/`):
  ```bash
  cd web && npm run build
  ```

---

## 📄 License

MIT License
