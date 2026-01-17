# MarketWise Lab Safety Monitoring System

AI-powered laboratory safety compliance monitoring using computer vision.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA GPU (recommended)
- [uv](https://docs.astral.sh/uv/) and [pnpm](https://pnpm.io/)

### Backend

```bash
cd backend
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

### Access

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Features

- **PPE Detection**: Goggles, masks, lab coats, gloves (YOLOv11)
- **Violation Detection**: Direct "No X" class detection
- **Person Segmentation**: SAM3 (preferred) / SAM2 (fallback)
- **Multi-Scale Detection**: Better small object detection
- **Temporal Filtering**: EMA confidence fusion
- **Person Tracking**: YOLOv8 native tracking
- **Face Recognition**: InsightFace for persistent identity

---

## Architecture

```
Frame → Person Detection → Segmentation → PPE Detection → Association
           (YOLOv8)        (SAM3/SAM2)      (YOLOv11)     (Mask-based)
                                                              ↓
                               ← Temporal Filter ← Tracking ←─┘
                               ↓
                        Event Persistence
```

### ML Pipeline

| Stage | Model | Purpose |
|-------|-------|---------|
| Person Detection | YOLOv8-medium | Detect persons with tracking |
| Segmentation | SAM3 / SAM2 | Generate person masks |
| PPE Detection | YOLOv11 | Detect PPE and violations |
| Face Recognition | InsightFace | Persistent identity |

---

## Model Setup

### YOLOv11 PPE Detector

Place your trained model at:
```
backend/weights/ppe_detector/best.pt
```

### SAM3 (Preferred)

Download via ModelScope (no authentication required):
```bash
pip install modelscope
modelscope download --model facebook/sam3 sam3/sam3.pt --local_dir backend/weights/sam3
```

Or place manually at: `backend/weights/sam3/sam3.pt`

### SAM2 (Fallback)

```bash
mkdir -p backend/weights/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
  -O backend/weights/sam2/sam2.1_hiera_base_plus.pt
```

### YOLOv8 Person Detector

Auto-downloads on first run.

---

## Configuration

Key settings in `.env`:

```env
# Segmentation (SAM3 preferred, SAM2 fallback)
USE_SAM3=true
SAM3_MODEL_PATH=weights/sam3/sam3.pt
USE_SAM2=true

# Detection
DETECTION_CONFIDENCE_THRESHOLD=0.5
VIOLATION_CONFIDENCE_THRESHOLD=0.3

# Multi-scale (for small objects like goggles)
MULTI_SCALE_ENABLED=true
MULTI_SCALE_FACTORS=[1.0, 1.5, 2.0]

# Temporal filtering
TEMPORAL_FUSION_STRATEGY=ema
TEMPORAL_EMA_ALPHA=0.7

# Required PPE (triggers violations if missing)
REQUIRED_PPE=["safety goggles", "face mask", "lab coat"]
```

---

## Project Structure

```
marketwise/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # API endpoints
│   │   ├── core/config.py    # Settings
│   │   ├── ml/               # ML pipeline
│   │   │   ├── pipeline.py
│   │   │   ├── hybrid_detector.py
│   │   │   ├── yolov11_detector.py
│   │   │   ├── sam3_segmenter.py
│   │   │   └── sam2_segmenter.py
│   │   ├── models/           # Database models
│   │   └── services/         # Business logic
│   ├── weights/              # Model weights
│   └── pyproject.toml
├── frontend/
│   ├── src/app/              # Next.js pages
│   └── src/components/       # React components
└── data/
    ├── videos/               # Input videos
    └── snapshots/            # Violation snapshots
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats/summary` | GET | Dashboard stats |
| `/api/events` | GET | Compliance events |
| `/api/persons` | GET | Tracked persons |
| `/api/stream/upload` | POST | Upload video |
| `/api/stream/live/feed` | GET | Live MJPEG stream |

---

## Demo Mode

```bash
# Mock detection (no models needed)
python demo.py --mock

# Process video
python demo.py --video data/videos/sample.mp4

# Webcam
python demo.py --camera 0
```

---

## Troubleshooting

### CUDA out of memory
```bash
CUDA_VISIBLE_DEVICES="" uv run uvicorn app.main:app --reload
```

### SAM3 not found
```bash
# Download via ModelScope
modelscope download --model facebook/sam3 sam3/sam3.pt --local_dir backend/weights/sam3
```

### No violations detected
- Lower `VIOLATION_CONFIDENCE_THRESHOLD` (default: 0.3)
- Enable `MULTI_SCALE_ENABLED=true`
- Check YOLOv11 model is loaded in logs

---

## Development

```bash
# Run tests
cd backend && uv run pytest

# Format code
uv run ruff format . && uv run ruff check . --fix

# Mock mode
USE_MOCK_DETECTOR=true uv run uvicorn app.main:app --reload
```

---

## License

Created for the MarketWise hackathon.
