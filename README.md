# MarketWise Lab Safety Monitoring System

AI-powered laboratory safety compliance monitoring using computer vision and deep learning.

---

## Table of Contents

- [Target Domain](#-target-domain-laboratory-safety)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [ML Pipeline](#ml-pipeline-deep-dive)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-endpoints)
- [Troubleshooting](#troubleshooting)

---

## Target Domain: Laboratory Safety

**Domain Choice**: This system is designed for **academic and research laboratory environments**.

### Laboratory vs Industrial PPE Requirements

| PPE Category              | Laboratory Environment | Industrial Environment | Our Implementation        |
| ------------------------- | ---------------------- | ---------------------- | ------------------------- |
| **Safety Goggles**        | Required               | Required               | Detected (71.6% mAP50)    |
| **Face Mask**             | Required               | Required               | Detected (80.5% mAP50)    |
| **Lab Coat**              | Required               | Varies                 | Detected (91.5% mAP50)    |
| **Gloves**                | Optional               | Required               | Detected (81.1% mAP50)    |
| **Protective Helmet/Cap** | Not required           | Required               | Not implemented           |
| **Safety Shoes**          | Not required           | Required               | Not implemented           |

**Extensibility**: The system can be extended for industrial use by retraining the YOLOv11 model with additional classes.

---

## Features

- **Real-time PPE Detection**: Detects safety goggles, masks, lab coats, and gloves
- **Multi-Scale Detection**: Improved small object (goggles) detection via multi-scale inference
- **Person Segmentation**: SAM3/SAM2 masks for accurate PPE-person association
- **Confidence Fusion**: Temporal EMA fusion for stable, accurate detections
- **Person Tracking**: YOLOv8 native tracking + DeepSORT for consistent identification
- **Face Recognition**: InsightFace (ArcFace) for persistent identity across sessions
- **Re-identification**: PersonGallery for re-ID across track deletions
- **Live Dashboard**: Real-time monitoring with violation statistics
- **Event Logging**: Complete audit trail with deduplication

---

## Architecture Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │           NEXT.JS FRONTEND              │
                                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
                                    │  │Live Feed│  │Dashboard│  │ Events  │  │
                                    │  │+Overlays│  │ +Stats  │  │  +Logs  │  │
                                    │  └─────────┘  └─────────┘  └─────────┘  │
                                    └──────────────────┬──────────────────────┘
                                                       │ REST / MJPEG
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                  FASTAPI BACKEND                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              ML PIPELINE                                        │  │
│  │                                                                                 │  │
│  │   Frame ─► Person Detection ─► Segmentation ─► PPE Detection ─► Association    │  │
│  │              (YOLOv8)          (SAM3/SAM2)      (YOLOv11)        (Mask-based)   │  │
│  │                                                                                 │  │
│  │         ─► Tracking ─► Face Recognition ─► Temporal Filter ─► Persistence      │  │
│  │           (DeepSORT)    (InsightFace)      (EMA Fusion)       (SQLite/PG)      │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline Deep Dive

### Detection Flow

```
Frame N
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. PERSON DETECTION (YOLOv8-medium)                        │
│     - Detects person bounding boxes                         │
│     - Native tracking provides consistent track_ids         │
│     Output: [{box, track_id, confidence}, ...]              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PERSON SEGMENTATION (SAM3 preferred, SAM2 fallback)     │
│     SAM3: Streaming video with automatic session management │
│     SAM2: Video propagation with explicit state management  │
│     Fallback: Box-based association if segmentation fails   │
│     Output: High-quality person masks per track_id          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. PPE & VIOLATION DETECTION (YOLOv11 custom-trained)      │
│     Multi-scale inference: 1x, 1.5x, 2x for small objects   │
│     NMS merges detections across scales                     │
│                                                             │
│     Classes:                                                │
│     ├─ PPE: Googles, Mask, Lab Coat, Gloves, Head Mask      │
│     ├─ Violations: No googles, No Mask, No Lab coat, etc.   │
│     └─ Actions: Drinking, Eating                            │
│                                                             │
│     Output: ppe_detections, violation_detections, actions   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. PPE-PERSON ASSOCIATION                                  │
│     Primary: Mask containment (if both masks available)     │
│     Fallback: Box containment, IoU, center-point checks     │
│                                                             │
│     For violations (small boxes):                           │
│     - Center inside person box OR                           │
│     - Containment >= 0.1 OR                                 │
│     - IoU >= 0.1                                            │
│                                                             │
│     Output: Each person gets detected_ppe, missing_ppe      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  5. TRACKING & FACE RECOGNITION                             │
│     DeepSORT: Kalman filter + Hungarian algorithm           │
│     InsightFace: 512-dim ArcFace embeddings                 │
│     PersonGallery: Re-ID across track deletions             │
│                                                             │
│     Output: Stable track_ids, person_ids (UUIDs)            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  6. TEMPORAL FILTERING & CONFIDENCE FUSION                  │
│     Buffer: 3 frames for stability                          │
│     Fusion: EMA (alpha=0.7) across frames                   │
│     Threshold: 0.4 after fusion                             │
│     Min frames: 2 consecutive for violation trigger         │
│                                                             │
│     Output: Stable, filtered violations                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  7. EVENT PERSISTENCE                                       │
│     Deduplication: Track active violations per person       │
│     Create event on violation start                         │
│     Update duration while ongoing                           │
│     Close event when violation ends                         │
│                                                             │
│     Output: ComplianceEvent records in database             │
└─────────────────────────────────────────────────────────────┘
```

### Segmentation Strategy (SAM3 vs SAM2)

| Feature | SAM3 (Preferred) | SAM2 (Fallback) |
|---------|------------------|-----------------|
| **API** | Single `process_frame()` call | Multiple calls (init, add, propagate) |
| **Session** | Auto-managed streaming | Manual state management |
| **Video Support** | Native streaming inference | Video propagation with intervals |
| **Model Source** | HuggingFace Transformers | Meta's sam2 package |
| **When Used** | `USE_SAM3=true` (default) | If SAM3 fails or `USE_SAM3=false` |

```python
# SAM3 - Simple API
masks = sam3_segmenter.process_frame(frame, persons)

# SAM2 - Multiple steps
sam2_segmenter.init_video_tracking(frame, persons)
sam2_segmenter.add_new_object(frame, box, track_id)
masks = sam2_segmenter.propagate_masks(frame)
```

### Multi-Scale Detection

For better small object (goggles) detection:

```
Original Frame (640x480)
    │
    ├─► Scale 1.0x ─► YOLOv11 ─► Detections A
    │
    ├─► Scale 1.5x ─► YOLOv11 ─► Detections B (scaled back)
    │
    └─► Scale 2.0x ─► YOLOv11 ─► Detections C (scaled back)
                                      │
                                      ▼
                               NMS Merge (IoU=0.5)
                                      │
                                      ▼
                              Final Detections
```

### Confidence Fusion

Temporal EMA fusion for stable detections:

```
confidence_t = α × detection_conf + (1-α) × confidence_{t-1}

Where:
  α = 0.7 (TEMPORAL_EMA_ALPHA)
  
Threshold: 0.4 (TEMPORAL_CONFIDENCE_THRESHOLD)
```

---

## Tech Stack

| Component        | Technology                                      |
| ---------------- | ----------------------------------------------- |
| Frontend         | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| Backend          | FastAPI, Python 3.11+, Pydantic                 |
| Person Detection | YOLOv8-medium (Ultralytics) with native tracking|
| PPE Detection    | YOLOv11 (custom trained on Safety Lab dataset)  |
| Segmentation     | SAM3 (preferred) / SAM2 (fallback)              |
| Face Recognition | InsightFace (ArcFace, buffalo_l)                |
| Tracking         | DeepSORT + PersonGallery re-identification      |
| Database         | SQLite (dev) / PostgreSQL (prod)                |
| Package Managers | `uv` (Python), `pnpm` (frontend)                |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (recommended)
- `uv` package manager: https://docs.astral.sh/uv/
- `pnpm` package manager: https://pnpm.io/

### Option 1: Demo Mode (No ML Models Required)

```bash
python demo.py --mock
```

### Option 2: Full Setup

#### 1. Backend Setup

```bash
cd backend

# Install uv if you haven't
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env as needed

# Run the server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup

```bash
cd frontend

# Install pnpm if you haven't
npm install -g pnpm

# Install dependencies
pnpm install

# Run development server
pnpm dev
```

#### 3. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Project Structure

```
marketwise/
├── backend/
│   ├── app/
│   │   ├── api/routes/           # API endpoints
│   │   │   ├── events.py         # Event CRUD
│   │   │   ├── persons.py        # Person management
│   │   │   ├── stats.py          # Statistics
│   │   │   └── stream.py         # Video processing
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic settings
│   │   │   └── database.py       # SQLAlchemy async
│   │   ├── ml/                   # ML Pipeline
│   │   │   ├── pipeline.py       # Main orchestration
│   │   │   ├── hybrid_detector.py # YOLOv11 + SAM3/SAM2
│   │   │   ├── yolov11_detector.py # PPE detection + multi-scale
│   │   │   ├── person_detector.py  # Person detection
│   │   │   ├── sam3_segmenter.py   # SAM3 streaming video
│   │   │   ├── sam2_segmenter.py   # SAM2 fallback
│   │   │   ├── tracker.py          # DeepSORT
│   │   │   ├── person_gallery.py   # Re-identification
│   │   │   ├── face_recognition.py # InsightFace
│   │   │   ├── temporal_filter.py  # EMA fusion
│   │   │   └── mask_utils.py       # Containment calculations
│   │   ├── models/               # Database models
│   │   │   ├── event.py          # ComplianceEvent
│   │   │   └── person.py         # Person
│   │   └── services/             # Business logic
│   │       ├── persistence.py    # Event persistence
│   │       └── deduplication.py  # Event deduplication
│   ├── weights/                  # Model weights
│   │   ├── ppe_detector/         # YOLOv11 weights
│   │   └── sam2/                 # SAM2 weights (if used)
│   ├── pyproject.toml            # Python dependencies
│   └── .env                      # Environment config
├── frontend/
│   ├── src/
│   │   ├── app/                  # Next.js pages
│   │   ├── components/           # React components
│   │   └── lib/                  # API client
│   └── package.json
├── data/
│   ├── videos/                   # Input videos
│   ├── processed/                # Processed videos
│   └── snapshots/                # Violation snapshots
├── demo.py                       # Demo script
├── AGENTS.md                     # AI agent instructions
├── ARCHITECTURE.md               # Detailed architecture
├── TRAINING_GUIDE.md             # Model training guide
└── README.md                     # This file
```

---

## Configuration

### Key Environment Variables

```env
# Detector Selection
DETECTOR_TYPE=hybrid              # hybrid, yolov11, mock

# SAM3 (Preferred Segmentation)
USE_SAM3=true                     # Enable SAM3 streaming video
SAM3_MODEL=facebook/sam3          # HuggingFace model

# SAM2 (Fallback Segmentation)
USE_SAM2=true                     # Fallback if SAM3 fails
SAM2_MODEL_TYPE=sam2.1_hiera_base_plus

# Multi-Scale Detection
MULTI_SCALE_ENABLED=true          # Better small object detection
MULTI_SCALE_FACTORS=[1.0, 1.5, 2.0]
MULTI_SCALE_NMS_THRESHOLD=0.5

# Temporal Filtering
TEMPORAL_FUSION_STRATEGY=ema      # ema, mean, max
TEMPORAL_EMA_ALPHA=0.7            # Weight for recent frames
TEMPORAL_CONFIDENCE_THRESHOLD=0.4

# Detection Thresholds
DETECTION_CONFIDENCE_THRESHOLD=0.5
VIOLATION_CONFIDENCE_THRESHOLD=0.3
MASK_CONTAINMENT_THRESHOLD=0.5

# Required PPE (violations for missing)
REQUIRED_PPE=["safety goggles", "face mask", "lab coat"]
```

See `.env.example` for full configuration options.

---

## API Endpoints

| Endpoint                        | Method    | Description                   |
| ------------------------------- | --------- | ----------------------------- |
| `/api/stats/summary`            | GET       | Dashboard statistics          |
| `/api/stats/timeline`           | GET       | Violation timeline            |
| `/api/events`                   | GET       | Compliance events (paginated) |
| `/api/events/recent-violations` | GET       | Recent violations             |
| `/api/persons`                  | GET       | Tracked individuals           |
| `/api/persons/{id}`             | PATCH     | Update person name            |
| `/api/stream/upload`            | POST      | Upload video for processing   |
| `/api/stream/process`           | POST      | Start video processing        |
| `/api/stream/jobs/{id}`         | GET       | Get job status                |
| `/api/stream/live/feed`         | GET       | MJPEG live webcam stream      |

---

## Demo Script

```bash
# Mock mode - no ML models needed
python demo.py --mock

# Webcam mode with mock detection
python demo.py --camera 0 --mock

# Process a video file
python demo.py --video data/videos/sample.mp4

# Save output video
python demo.py --mock --output demo_output.mp4
```

**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `SPACE` - Pause (video mode)

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Use CPU mode
CUDA_VISIBLE_DEVICES="" uv run uvicorn app.main:app --reload

# Or reduce frame rate
FRAME_SAMPLE_RATE=5
```

### "SAM3 model requires authentication"

```bash
# Login to HuggingFace
uv run huggingface-cli login
```

### "Model not found"

```bash
# Check model path
ls backend/weights/ppe_detector/

# Use mock mode for testing
USE_MOCK_DETECTOR=true uv run uvicorn app.main:app --reload
```

### "WebSocket connection failed"

- Check backend is running on port 8000
- Verify `NEXT_PUBLIC_API_URL=http://localhost:8000` in frontend
- Check CORS settings in backend config

### "No violations detected"

- Check `VIOLATION_CONFIDENCE_THRESHOLD` (lower = more sensitive)
- Verify YOLOv11 model is loaded (check logs)
- Try `MULTI_SCALE_ENABLED=true` for better small object detection

---

## Development

### Running Tests

```bash
cd backend
uv run pytest
```

### Code Formatting

```bash
cd backend
uv run ruff format .
uv run ruff check . --fix
```

### Mock Mode

```bash
USE_MOCK_DETECTOR=true USE_MOCK_FACE=true uv run uvicorn app.main:app --reload
```

---

## Performance Tips

1. **GPU**: Use CUDA-capable GPU for best performance
2. **Frame Rate**: Reduce `FRAME_SAMPLE_RATE` (default: 10) for faster processing
3. **Multi-Scale**: Disable if not detecting small objects (`MULTI_SCALE_ENABLED=false`)
4. **Segmentation**: SAM3 is faster than SAM2 for video
5. **Model Size**: Use `sam2.1_hiera_tiny` for faster SAM2 if needed

---

## License

This project was created for the MarketWise hackathon.
