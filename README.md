# MarketWise Lab Safety Monitoring System

AI-powered laboratory safety compliance monitoring using computer vision and deep learning.

---

## 🎯 Target Domain: Laboratory Safety

**Domain Choice**: This system is designed for **academic and research laboratory environments**.

The problem statement requires teams to "choose and clearly specify their target domain." We have selected **laboratory safety** as our primary domain, focusing on PPE requirements typical in academic and research lab settings.

### Laboratory vs Industrial PPE Requirements

| PPE Category              | Laboratory Environment | Industrial Environment | Our Implementation        |
| ------------------------- | ---------------------- | ---------------------- | ------------------------- |
| **Safety Goggles**        | ✅ Required            | ✅ Required            | ✅ Detected (71.6% mAP50) |
| **Face Mask**             | ✅ Required            | ✅ Required            | ✅ Detected (80.5% mAP50) |
| **Lab Coat**              | ✅ Required            | ⚠️ Varies              | ✅ Detected (91.5% mAP50) |
| **Protective Helmet/Cap** | ❌ Not required        | ✅ Required            | ❌ Not implemented        |
| **Safety Shoes**          | ❌ Not required        | ✅ Required            | ❌ Not implemented        |
| **Gloves**                | ⚠️ Optional            | ✅ Required            | ✅ Detected (81.1% mAP50) |

**Rationale for Domain Selection:**

- Laboratory environments have distinct PPE requirements compared to industrial settings
- Focus on academic/research labs allows for specialized detection models
- Helmet and safety shoes are not standard requirements in lab environments
- The system can be extended for industrial use with additional training data

**Extensibility**: The system architecture supports adding industrial PPE detection. To enable helmet/shoes detection:

1. Source or collect training data for these classes
2. Retrain YOLOv11 model with new classes
3. Update `REQUIRED_PPE` in `backend/app/core/config.py`
4. Update detection prompts in configuration

---

## PPE Detection Scope

| PPE Item       | Detected | Required (Lab) | mAP50 | Notes                            |
| -------------- | -------- | -------------- | ----- | -------------------------------- |
| Safety Goggles | ✅ Yes   | ✅ Yes         | 71.6% | Acceptable for demo              |
| Face Mask      | ✅ Yes   | ✅ Yes         | 80.5% | Good performance                 |
| Lab Coat       | ✅ Yes   | ✅ Yes         | 91.5% | Excellent performance            |
| Gloves         | ✅ Yes   | Optional       | 81.1% | Detected but not enforced        |
| Head Mask      | ✅ Yes   | Optional       | N/A   | Detected but not enforced        |
| Helmet/Cap     | ❌ No    | ❌ No          | N/A   | Not required in lab environments |
| Safety Shoes   | ❌ No    | ❌ No          | N/A   | Not required in lab environments |

> **Note**: For industrial environments (factories, construction sites), helmet and safety shoes detection would require additional training data. The current model is optimized for laboratory safety compliance.

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
| PPE Detection    | SAM 3 (Meta) / YOLOv11 (Ultralytics)            |
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

## Problem Statement Compliance

This section maps the hackathon problem statement requirements to our implementation:

### Functional Requirements

| Requirement                     | Status         | Implementation Details                                  |
| ------------------------------- | -------------- | ------------------------------------------------------- |
| **Input: Live webcam feed**     | ✅ Implemented | MJPEG endpoint at `/api/stream/live/feed`               |
| **Input: Pre-recorded video**   | ✅ Implemented | Full upload + processing pipeline                       |
| **Detect human presence**       | ✅ Implemented | YOLOv8 person detection with tracking                   |
| **Detect helmet/cap**           | ⚠️ Domain N/A  | Not required in lab settings (see domain specification) |
| **Detect safety shoes**         | ⚠️ Domain N/A  | Not required in lab settings (see domain specification) |
| **Detect goggles/specs**        | ✅ Implemented | YOLOv11 trained model (71.6% mAP50)                     |
| **Detect mask**                 | ✅ Implemented | YOLOv11 trained model (80.5% mAP50)                     |
| **Detect lab coat**             | ✅ Implemented | YOLOv11 trained model (91.5% mAP50)                     |
| **Associate with individuals**  | ✅ Implemented | Face recognition (InsightFace) + tracking (DeepSORT)    |
| **Handle multiple individuals** | ✅ Implemented | DeepSORT multi-object tracking                          |
| **Video feed display**          | ✅ Implemented | VideoPlayer component with streaming                    |
| **Visual indicators/overlays**  | ✅ Implemented | Bounding boxes + mask overlays                          |
| **Admin dashboard**             | ✅ Implemented | Next.js dashboard with tabs                             |
| **Detection logs**              | ✅ Implemented | Events table with filtering                             |
| **Compliance statistics**       | ✅ Implemented | Stats cards + charts                                    |
| **Date/time of detection**      | ✅ Implemented | Timestamps on all events                                |
| **Store compliance records**    | ✅ Implemented | SQLite with full event model                            |
| **Person identifier**           | ✅ Implemented | Face embeddings + UUIDs                                 |
| **Detected safety equipment**   | ✅ Implemented | JSON array in events                                    |
| **Missing safety equipment**    | ✅ Implemented | JSON array in events                                    |
| **Timestamp**                   | ✅ Implemented | DateTime field                                          |
| **Video/camera source**         | ✅ Implemented | String field in events                                  |

**Legend**: ✅ Fully Implemented | ⚠️ Partially Implemented / Domain N/A | ❌ Not Implemented

### Detector Selection

Configure via `DETECTOR_TYPE` in `.env`:

- `hybrid` - YOLOv11 + SAM2 hybrid (recommended, best accuracy)
- `YOLOv11` - Use trained YOLOv11 model (faster, no masks)
- `sam3` - Use SAM 3 text-prompted detection
- `mock` - Mock detector for development

See [YOLOv11_SETUP.md](YOLOv11_SETUP.md) for integration instructions.

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

# Detector Selection: "sam3", "YOLOv11", or "mock"
DETECTOR_TYPE=YOLOv11

# YOLOv11 Model Path (relative to weights/ or absolute)
# Will auto-detect best.onnx or best.pt in weights/ppe_detector/ if not set
YOLOv11_MODEL_PATH=weights/ppe_detector/best.onnx

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
- Use CPU mode: `CUDA_VISIBLE_DEVICES="" uv run uvicorn app.main:app --reload`
- Reduce `FRAME_SAMPLE_RATE` in config (e.g., from 10 to 5)

### "Module not found"

- Ensure you're in the virtual environment
- Run `uv sync` to install dependencies
- Check that all required packages are in `pyproject.toml`

### "Model not found"

- Check `YOLOV11_MODEL_PATH` in `.env` file
- Ensure model file exists at `backend/weights/ppe_detector/best.pt` or `best.onnx`
- Use mock mode for testing: `USE_MOCK_DETECTOR=true`

### "WebSocket connection failed" / "Failed to connect to backend"

- Check backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local`
- Check CORS settings in backend config
- Verify firewall isn't blocking connections

### "Video processing failed"

- Check video format is supported (mp4, avi, mov, mkv, webm)
- Verify video file is not corrupted
- Check backend logs for detailed error messages
- Try converting video to MP4: `ffmpeg -i input.webm -c:v libx264 output.mp4`
- Use mock mode to isolate ML issues: `USE_MOCK_DETECTOR=true USE_MOCK_FACE=true`

### "Database error"

- Check `DATABASE_URL` in `.env` file
- Ensure database file/directory is writable
- Check SQLite version compatibility

### "Live webcam feed not working"

- Check webcam is connected and accessible
- Verify webcam permissions (Windows: Privacy settings)
- Try different webcam index (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)
- Check backend logs for webcam initialization errors

### Use Mock Mode for Development

For testing without ML models:

```bash
cd backend
USE_MOCK_DETECTOR=true USE_MOCK_FACE=true uv run uvicorn app.main:app --reload
```

This allows you to test the full system flow without requiring model weights.

## Future Enhancements

The following features have infrastructure in place but are not yet fully integrated:

### SAM2 Video Propagation

The system includes SAM2 (Segment Anything Model 2) infrastructure for temporal mask tracking across video frames. Currently using per-frame box-prompted segmentation which works correctly. Video propagation integration would:

- Improve segmentation consistency across frames
- Reduce computational overhead by propagating masks
- Enable smoother mask transitions during tracking

**Status:** Infrastructure exists in `backend/app/ml/sam2_segmenter.py`. Integration with the main pipeline is a future enhancement.

### Live Webcam Streaming

✅ **Implemented**: MJPEG live streaming endpoint at `/api/stream/live/feed` with real-time detection annotations. The system processes live webcam feeds with the same detection pipeline used for recorded videos.

### Event Deduplication

Currently every frame with a violation creates an event. Future enhancement could aggregate continuous violations into single events with start/end timestamps.

## License

This project was created for the MarketWise hackathon.
