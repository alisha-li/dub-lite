# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dub-lite** is a "bring your own model" AI-powered video dubbing platform for language learners. Users upload a video or provide a URL, select a target language and translation provider, and receive a dubbed video.

## Development Commands

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev       # Start dev server
npm run build     # Production build
npm run lint      # ESLint
npm run preview   # Preview production build
```

### Backend API (FastAPI)
```bash
cd api
pip install -r ../requirements.txt
uvicorn app:app --reload --port 8000
```

### Pipeline (Modal serverless)
```bash
# Deploy pipeline to Modal
modal deploy pipeline/modal_pipeline.py

# Run pipeline locally (for testing)
cd pipeline
python main.py
```

## Architecture

The app has three components:

### Frontend (`frontend/`)
Single-page React 19 + Vite app. No TypeScript. Key behaviors:
- Gets a presigned URL from the API, then uploads video directly to DigitalOcean Spaces (client never routes through API server)
- Polls `/api/jobs/{job_id}` every 2 seconds for progress updates
- Supports multiple translation providers: Groq, Gemini, Helsinki (free/offline)

### API (`api/app.py`)
FastAPI service that:
- Issues presigned PUT URLs for direct client-to-S3 uploads
- Creates jobs in Redis, then spawns a Modal serverless function to process them
- Exposes job status/progress (read from Redis + Modal Dict) for frontend polling
- Issues presigned GET URLs for video downloads

**State**: Redis (5-day TTL) stores job metadata and progress. No traditional database.

### Pipeline (`pipeline/`)
The core AI processing logic, runs on Modal with a T4 GPU:

**Orchestrator**: `pipeline/main.py` — `YTDubPipeline.dub()` runs these stages in order:
1. Download & audio extraction (yt-dlp + ffmpeg)
2. Speaker diarization (PyAnnote)
3. Transcription (Faster-Whisper, medium model)
4. Speaker-to-segment assignment
5. Sentence segmentation (NLTK)
6. Translation (Groq / Gemini / MarianMT)
7. Emotion classification (SpeechBrain wav2vec2)
8. Text-to-speech (Coqui TTS v2, multilingual)
9. Audio stitching with background noise overlay
10. Video reencoding (ffmpeg)
11. Upload to DigitalOcean Spaces

**Utilities**: `pipeline/utils.py` — all the heavy lifting (769 lines). Each stage above maps to one or more functions here.

**Modal wrapper**: `pipeline/modal_pipeline.py` — defines the Modal image (custom Docker with PyTorch, Whisper, TTS, etc.), GPU type (T4), timeout (1 hour), and volume for model caching.

## Infrastructure

| Service | Purpose |
|---|---|
| DigitalOcean Spaces | S3-compatible video storage |
| Redis | Job state and progress tracking |
| Modal | Serverless GPU execution |
| HuggingFace | Model hosting (PyAnnote, Whisper, MarianMT) |
| Groq | Fast LLM inference for translation |
| Google Gemini | High-quality translation |

## Environment Variables

The API and pipeline expect:
- `HF_TOKEN` — HuggingFace token (for PyAnnote models)
- `GROQ_API_KEY` — Groq API key
- `PYANNOTE_API_KEY` — PyAnnote AI SDK key
- `GEMINI_API_KEY` — Google Gemini key
- `SPACES_ACCESS_KEY`, `SPACES_SECRET_KEY`, `SPACES_ENDPOINT`, `SPACES_REGION`, `SPACES_BUCKET` — DigitalOcean Spaces credentials

## Key Design Decisions

- **No server-side upload**: Videos go client → Spaces directly via presigned URLs. The API server never handles video bytes.
- **Progress reporting**: Pipeline stages report `(stage_name, percent)` tuples into a Modal shared Dict; API reads and relays to frontend.
- **Translation context**: Translation functions pass previous/next sentences for context-aware output.
- **Emotion-aware TTS**: SpeechBrain classifies emotion per segment; Coqui TTS uses this for more natural speech.
- **Provider fallback**: Helsinki/MarianMT is always available as a free, offline translation option when no API keys are provided.
