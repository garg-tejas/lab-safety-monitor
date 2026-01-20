# SentinelVision

Real-time PPE compliance monitoring for laboratories and industrial spaces.

SentinelVision is an event-driven video processing system that turns noisy, frame-level PPE detections into stable violation events (start / end / duration / evidence) under real-world constraints like occlusion, limited compute, and long-running streams.

The focus is stability under real-world constraints: occlusion, limited processing FPS, false positives, and long-running streams.

---

## Quick overview

- Problem: monitoring does not scale; per-frame alerts create noise under occlusion
- Constraints: real-time video, limited compute, multiple people, low tolerance for false alerts
- System loop: input (video) -> processing (detect/associate/filter) -> output (events + UI updates)
- Trade-offs: stable event tracking over per-frame accuracy
- Limitations: face recognition and multi-scale settings depend on image quality and GPU budget
- Run it: see Setup and Usage

---

## Problem

Safety enforcement in labs and industrial spaces breaks down when monitoring is manual, inconsistent, or reduced to frame-by-frame alerts.

Common failure modes:
- Manual monitoring does not scale and varies by operator
- Conventional CCTV has no notion of PPE compliance
- Per-frame PPE demos create alert spam under occlusion and partial visibility

---

## Constraints

This project assumes:
- Real-time video input (webcam + recorded footage)
- Limited processing budget (cannot run heavy models on every frame)
- Frequent occlusion (hands, motion blur, partial visibility)
- Multiple people in frame
- Low tolerance for false alerts

---

## Overview

Pipeline (high level):
1. Video stream (live or file)
2. Person detection + tracking
3. Segmentation for person masks (used for PPE-to-person association)
4. PPE/violation detection
5. Temporal filtering across frames
6. Event creation + duration tracking
7. Dashboard updates (stream + events)

When a violation is detected, the system treats it as an event state that can start, continue, and end, rather than a one-off prediction. This shifts the system from frame-level prediction to a stateful, event-driven model.


---

## Key design choices

Hybrid detection pipeline (lower association errors):
- YOLOv8 for person detection + tracking
- SAM3 for person masks
- YOLOv11 for PPE items and violation classes
- Cost: more moving parts and more compute than a single model

Temporal filtering (reduce alert spam):
- EMA confidence fusion across frames
- Hysteresis thresholds for starting/stopping violations
- Subset matching during short occlusions
- Cost: alerts can be delayed by a few frames

Streaming (predictable latency):
- Display FPS and ML processing FPS are decoupled
- Frames can be cached and dropped to keep processing stable under load
- Cost: some frames are not processed

Event-centric storage (audit-friendly output):
- Start frame, end frame, duration, evidence snapshot
- Associated identity when available

---

## Limitations

- Face recognition degrades under heavy occlusion or low resolution
- Multi-scale PPE detection increases GPU memory usage
- SQLite is used for local-first development and is not intended for horizontal scaling

---

## What this demonstrates

- Designing an ML pipeline around operational constraints (not best-case inputs)
- Reasoning about trade-offs: accuracy vs latency vs reliability
- Running long-lived video processing with backpressure (drop/caching) instead of building a one-off demo
- Shipping an end-to-end loop (stream -> detection -> event storage -> UI)

---

## Stack

- Backend: FastAPI (async)
- Frontend: Next.js 15, Tailwind CSS
- ML: YOLOv8, YOLOv11 (custom), SAM3, InsightFace
- Storage: SQLite (async)
- Streaming: MJPEG + WebSockets

---

## Quick start

```bash
git clone https://github.com/garg-tejas/lab-safety-monitor.git
cd lab-safety-monitor
```

---

## Setup
- Prereqs: Python 3.11+, Node.js 18+ (pnpm), `uv` (CUDA GPU recommended)

Backend (FastAPI):

```bash
cd backend
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (Next.js):

```bash
cd frontend
pnpm install
pnpm dev
```

URLs:
- UI: http://localhost:3000
- API docs: http://localhost:8000/docs

Model weights:
- SAM3: `backend/weights/sam3/sam3.pt`
- YOLOv11 PPE: `backend/weights/ppe_detector/best.pt`
- If you don't have SAM3 weights, disable SAM3 via env or download weights before running.

---

## Usage

The API is easiest to explore via `http://localhost:8000/docs`. Examples:

### Process a video file

```bash
curl -X POST http://localhost:8000/api/stream/upload \
  -F "file=@lab_demo.mp4"

curl -X POST http://localhost:8000/api/stream/process \
  -H "Content-Type: application/json" \
  -d '{"video_path": "lab_demo.mp4", "save_output": true}'
```

### Start/stop live monitoring

```bash
curl -X POST http://localhost:8000/api/stream/live/start
curl -X POST http://localhost:8000/api/stream/live/stop
```

---

## Project structure

- `backend/app/api/routes/`: REST endpoints
- `backend/app/ml/`: ML pipeline
- `backend/app/services/`: persistence + deduplication
- `backend/weights/`: model weights
- `frontend/src/app/`: pages
- `frontend/src/components/`: UI components
- `data/`: videos, processed outputs, snapshots

---

## Configuration

Copy `backend/.env.example` to `backend/.env`, then adjust the relevant settings:

- `REQUIRED_PPE`
- `DETECTION_CONFIDENCE_THRESHOLD`, `VIOLATION_CONFIDENCE_THRESHOLD`
- `TEMPORAL_FUSION_STRATEGY`, `TEMPORAL_EMA_ALPHA`, `TEMPORAL_*_MIN_FRAMES*`
- `LIVE_STREAM_DISPLAY_FPS`, `LIVE_STREAM_PROCESS_FPS`, `LIVE_STREAM_QUEUE_SIZE`
- `MULTI_SCALE_ENABLED`, `MULTI_SCALE_FACTORS`
- `USE_SAM3`, `USE_SAM2`
- `DATABASE_URL`
- `ENABLE_SNAPSHOT_CAPTURE`, `SNAPSHOTS_DIR`

---

## API

OpenAPI docs are available at `http://localhost:8000/docs` when the backend is running.

---

## ML models

### Custom YOLOv11 PPE detector

Trained on a lab safety dataset with 12 detection classes:

- PPE items: `goggles`, `mask`, `lab_coat`, `gloves`, `head_mask`
- Violations: `no_goggles`, `no_mask`, `no_lab_coat`, `no_gloves`, `no_head_mask`
- Actions: `drinking`, `eating`

### Pre-trained Models

- YOLOv8-medium (person detection + tracking)
- SAM3 (segmentation), SAM2 fallback
- InsightFace (ArcFace) for face embeddings

---

## Troubleshooting

### CUDA Out of Memory

- Symptoms: `RuntimeError: CUDA out of memory`

- Fixes:
```bash
# Option 1: Force CPU mode
CUDA_VISIBLE_DEVICES="" uv run uvicorn app.main:app --reload

# Option 2: Reduce processing scale
# In .env: MULTI_SCALE_FACTORS=[1.0]

# Option 3: Lower processing FPS
# In .env: LIVE_STREAM_PROCESS_FPS=1
```

### No Violations Detected

- Symptoms: dashboard shows 0 violations despite visible missing PPE

- Fixes:
1. Lower confidence threshold in `.env`:
   ```env
   VIOLATION_CONFIDENCE_THRESHOLD=0.2  # Default: 0.3
   ```

2. Enable multi-scale detection for small objects:
   ```env
   MULTI_SCALE_ENABLED=true
   ```

3. Verify the YOLOv11 model is loaded correctly:
   - Check logs for `Loaded YOLOv11 PPE detector`
   - Ensure `weights/ppe_detector/best.pt` exists

4. Check required PPE configuration:
   ```env
   REQUIRED_PPE=["safety_goggles", "face_mask", "lab_coat"]
   ```

### Live Stream Lag / Stuttering

- Symptoms: live feed freezes or has significant delay

- Fixes:
1. Reduce processing FPS (default: 3):
   ```env
   LIVE_STREAM_PROCESS_FPS=2  # Or even 1
   ```

2. Decrease queue size (default: 2):
   ```env
   LIVE_STREAM_QUEUE_SIZE=1
   ```

3. Verify GPU utilization:
   ```bash
   nvidia-smi
   ```

4. Disable multi-scale detection for faster processing:
   ```env
   MULTI_SCALE_ENABLED=false
   ```

### Frontend Not Connecting to Backend

- Symptoms: dashboard shows “Connection Error” or empty data

- Fixes:
1. Verify backend is running:
   ```bash
   curl http://localhost:8000/docs  # Should return Swagger UI
   ```

2. Check CORS configuration in `backend/app/main.py`:
   ```python
   allow_origins=["http://localhost:3000"]
   ```

3. Update API URL in `frontend/.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

---

## Acknowledgments

- YOLOv8/v11: Ultralytics ([https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics))
- SAM3: Meta AI ([https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2))
- InsightFace: DeepInsight ([https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface))
- FastAPI: Sebastián Ramírez ([https://fastapi.tiangolo.com](https://fastapi.tiangolo.com))
- Next.js: Vercel ([https://nextjs.org](https://nextjs.org))