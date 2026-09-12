# Multi-Modal Educational Intelligence Copilot: Data Ingestion (Team 4a)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)
![Celery](https://img.shields.io/badge/Celery-Task%20Queue-brightgreen)

The **Multi-Modal Data Ingestion Subsystem** is Team 4a's core component of the **Multi-Modal Educational Intelligence Copilot**. It processes heterogeneous educational content (**PDF documents**, **MP4 lecture videos**, and **YouTube URLs**), normalizes text, extracts video keyframes and lecture slide text via OCR, computes dense vector embeddings, and publishes enriched knowledge vectors into **Qdrant** for downstream RAG retrieval by **Team 4b**.

---

## 🌟 Key Features & Novelties

1. **Multi-Source Ingestion Pipeline**:
   - **PDFs**: Structure-aware reading order extraction via `PyMuPDF`, hierarchical heading detection, and image-only page skipping.
   - **MP4 Videos**: Audio track isolation with `ffmpeg`, transcription via `Whisper` with $\pm 1$s timestamp accuracy, and `audio_absent` resilience.
   - **YouTube URLs**: Timed transcript fetching via `youtube-transcript-api` and video metadata parsing via `yt-dlp`.
2. **Novelty 1: Slide OCR & Visual Grounding**:
   - High-resolution keyframe snapshots are parsed with pure-Python/ONNX `RapidOCR` to capture mathematical formulas, diagrams, and bullet points on lecture slides.
3. **Novelty 3: Table-to-Markdown Extraction**:
   - Academic and syllabus tables in PDFs are detected and converted into Markdown tables (`| ... |`) to preserve row/column semantic relationships.
4. **Novelty 4: Dynamic Scene-Change Detection**:
   - Evaluates frame differences in lecture videos to capture keyframes when lecture slides flip, rather than relying on a rigid fixed timer.
5. **Robust Pipeline Architecture**:
   - Sentence-boundary-preserving token chunker (512 tokens, 50 overlap).
   - L2-normalized 384-dimensional dense embeddings (`all-MiniLM-L6-v2`).
   - Qdrant batch publishing with exponential backoff ($2\text{s}, 4\text{s}, 8\text{s}$).
   - Asynchronous job management with Celery and stage progress reporting ($25\% \to 50\% \to 75\% \to 100\%$).

---

## 🚀 Quickstart Guide

### Option A: Lightweight Local Mode (Zero Docker required)

Ideal for rapid local development and testing without starting a Docker daemon or Redis service:

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Configure Environment**:
   Ensure `.env` contains:
   ```ini
   QDRANT_MODE=embedded
   CELERY_ALWAYS_EAGER=true
   ```

3. **Start FastAPI Development Server**:
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Access Interactive API Docs**:
   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### Option B: Production Mode via Docker Compose

Deploys the full stack with FastAPI, 10-worker Celery pool, Redis broker, and Qdrant container:

```bash
docker compose up --build
```

Endpoints available:
- FastAPI API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ingest/pdf` | Upload PDF file (multipart/form-data) |
| `POST` | `/ingest/video` | Upload MP4 video file (multipart/form-data) |
| `POST` | `/ingest/youtube` | Submit YouTube URL `{"url": "...", "language": "en"}` |
| `GET` | `/jobs/{job_id}` | Retrieve job progress (0% - 100%), status, and errors |
| `GET` | `/jobs` | Paginated list of all ingestion jobs |
| `GET` | `/health` | Health check and subsystem configuration |

---

## 🧪 Running Automated Tests

Run the complete test suite verifying all 29 correctness properties:

```powershell
pytest -v
```

---

## 🤝 Handoff for Team 4b
See [TEAM_4B_HANDOFF.md](file:///d:/Avdhut/programing/Accion%20labs%20project/TEAM_4B_HANDOFF.md) for the complete collection schema, metadata attributes, and sample query code.

