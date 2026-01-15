# MarketWise Lab Safety Monitoring System

AI-powered laboratory safety compliance monitoring using computer vision and deep learning.

## Target Domain: Laboratory Safety

This system is designed for **academic and research laboratory environments** where the following PPE is required:

- **Safety goggles** - Protective eyewear
- **Face masks** - Respiratory protection
- **Lab coats** - Body protection

**Note:** Industrial PPE (hard hats, safety shoes) is not required in typical laboratory settings and is therefore not enforced by default. The system can be reconfigured for industrial use by modifying `REQUIRED_PPE` in `backend/app/core/config.py`.

## Features

- **Real-time PPE Detection**: Detects safety goggles, masks, and lab coats
- **Person Tracking**: DeepSORT multi-object tracking for consistent person identification
- **Face Recognition**: InsightFace (ArcFace) for persistent identity across sessions
- **Temporal Filtering**: 3-frame buffer to prevent flickering false alerts
- **Live Dashboard**: Real-time monitoring with violation statistics
- **Event Logging**: Complete audit trail of compliance events

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NEXT.JS FRONTEND                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Live Feed   │  │ Dashboard   │  │ Events Log      │  │
│  │ + Overlays  │  │ + Stats     │  │ + Filters       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │ REST / WebSocket
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ML PIPELINE                         │   │
│  │  Frame → Detection → Tracking → Face ID → Log   │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│                    ┌─────▼─────┐                       │
│                    │  SQLite   │                       │
│                    └───────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component        | Technology                                      |
| ---------------- | ----------------------------------------------- |
| Frontend         | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| Backend          | FastAPI, Python 3.11+                           |
| PPE Detection    | SAM 3 (Meta) / YOLOv8 (Ultralytics)             |
| Face Recognition | InsightFace (ArcFace)                           |
| Tracking         | DeepSORT (Kalman + Hungarian)                   |
| Database         | SQLite (dev) / PostgreSQL (prod)                |
| Package Managers | `uv` (Python), `pnpm` (frontend)                |

## Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (recommended)
- `uv` package manager: https://docs.astral.sh/uv/
- `pnpm` package manager: https://pnpm.io/

## Quick Start

### Option 1: Demo Mode (No ML Models Required)

```bash
# Run the demo with mock detections
python demo.py --mock
```

This runs a simulated lab scene showing the detection pipeline without needing ML models.

### Option 2: Full Setup

#### 1. Backend Setup

```bash
cd backend

# Install uv if you haven't
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

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

## Project Structure

```
marketwise/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # API endpoints
│   │   ├── core/            # Config & database
│   │   ├── ml/              # ML pipeline
│   │   │   ├── sam3_detector.py    # PPE detection
│   │   │   ├── face_recognition.py # Face ID
│   │   │   ├── tracker.py          # DeepSORT
│   │   │   ├── temporal_filter.py  # Flickering prevention
│   │   │   └── pipeline.py         # Main orchestration
│   │   ├── models/          # Database models
│   │   └── main.py
│   ├── pyproject.toml       # uv dependencies
│   └── weights/             # Model weights folder
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   ├── components/      # React components
│   │   └── lib/             # API client
│   ├── package.json
│   └── .npmrc               # pnpm config
├── data/
│   └── videos/              # Sample videos
├── demo.py                  # Demo script
├── TRAINING_GUIDE.md        # Model training instructions
└── README.md
```

## API Endpoints

| Endpoint                        | Method    | Description                   |
| ------------------------------- | --------- | ----------------------------- |
| `/api/stats/summary`            | GET       | Dashboard statistics          |
| `/api/stats/timeline`           | GET       | Violation timeline            |
| `/api/events`                   | GET       | Compliance events (paginated) |
| `/api/events/recent-violations` | GET       | Recent violations             |
| `/api/persons`                  | GET       | Tracked individuals           |
| `/api/stream/live`              | WebSocket | Live video stream             |

## Demo Script

The demo script allows testing without a full setup:

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

## PPE Detection

The system detects PPE items relevant to laboratory environments:

| PPE Item          | Detection Method | Status                   | Required |
| ----------------- | ---------------- | ------------------------ | -------- |
| Safety Goggles    | YOLOv8 (trained) | ✅ YOLOv8: 71.6% mAP50   | Yes      |
| Face Mask         | YOLOv8 (trained) | ✅ YOLOv8: 80.5% mAP50   | Yes      |
| Lab Coat          | YOLOv8 (trained) | ✅ YOLOv8: 91.5% mAP50   | Yes      |
| Protective Helmet | Not implemented  | N/A (not needed in labs) | No       |
| Safety Shoes      | Not implemented  | N/A (not needed in labs) | No       |

**Detector Selection**: Configure via `DETECTOR_TYPE` in `.env`:

- `hybrid` - YOLOv8 + SAM2 hybrid (recommended, best accuracy)
- `yolov8` - Use trained YOLOv8 model (faster, no masks)
- `sam3` - Use SAM 3 text-prompted detection
- `mock` - Mock detector for development

See [YOLOV8_SETUP.md](YOLOV8_SETUP.md) for integration instructions.

## Configuration

### Backend Environment Variables

Create `backend/.env`:

```env
# Application
DEBUG=True
APP_NAME=MarketWise Lab Safety

# Database
DATABASE_URL=sqlite+aiosqlite:///./marketwise.db

# ML Settings
DETECTION_CONFIDENCE_THRESHOLD=0.5
FACE_RECOGNITION_THRESHOLD=0.6
USE_MOCK_DETECTOR=false
USE_MOCK_FACE=false

# Detector Selection: "sam3", "yolov8", or "mock"
DETECTOR_TYPE=yolov8

# YOLOv8 Model Path (relative to weights/ or absolute)
# Will auto-detect best.onnx or best.pt in weights/ppe_detector/ if not set
YOLOV8_MODEL_PATH=weights/ppe_detector/best.onnx

# Video Processing
FRAME_SAMPLE_RATE=10
TEMPORAL_BUFFER_SIZE=3
```

### Frontend Environment Variables

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Model Weights

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions on:

- Training models on Kaggle
- Downloading pre-trained weights
- Creating custom PPE datasets

### Quick Weight Setup

```bash
# Create weights directory
mkdir -p backend/weights/{ppe_detector,sam3,face_recognition}

# InsightFace downloads automatically on first run
# SAM 3 weights need to be downloaded from Meta
```

## Development

### Mock Mode

For development without ML models:

```bash
# Backend with mock detectors
cd backend
USE_MOCK_DETECTOR=true USE_MOCK_FACE=true uv run uvicorn app.main:app --reload
```

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

## Troubleshooting

### "CUDA out of memory"

- Reduce batch size in config
- Use CPU mode: `CUDA_VISIBLE_DEVICES="" python demo.py`

### "Module not found"

- Ensure you're in the virtual environment
- Run `uv sync` to install dependencies

### "WebSocket connection failed"

- Check backend is running on port 8000
- Check CORS settings in config

## Future Enhancements

The following features have infrastructure in place but are not yet fully integrated:

### SAM2 Video Propagation

The system includes SAM2 (Segment Anything Model 2) infrastructure for temporal mask tracking across video frames. Currently using per-frame box-prompted segmentation which works correctly. Video propagation integration would:

- Improve segmentation consistency across frames
- Reduce computational overhead by propagating masks
- Enable smoother mask transitions during tracking

**Status:** Infrastructure exists in `backend/app/ml/sam2_segmenter.py`. Integration with the main pipeline is a future enhancement.

### Live Webcam Streaming

MJPEG live streaming endpoint and frontend player are planned for real-time monitoring scenarios. Currently the system processes pre-recorded videos which is sufficient for most demo and deployment scenarios.

### Event Deduplication

Currently every frame with a violation creates an event. Future enhancement could aggregate continuous violations into single events with start/end timestamps.

## License

This project was created for the MarketWise hackathon.
