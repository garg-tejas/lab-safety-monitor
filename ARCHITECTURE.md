# MarketWise Lab Safety System - Architecture

## System Overview

The MarketWise Lab Safety Monitoring System is an end-to-end AI-powered solution for real-time safety compliance monitoring in laboratory environments. The system uses computer vision and machine learning to detect PPE violations and associate them with specific individuals.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND LAYER                                  │
│                           (Next.js 16 + TypeScript)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Live Video     │  │  Dashboard      │  │  Events & Analytics         │  │
│  │  + MJPEG Stream │  │  + Stats Cards  │  │  + Filtering                │  │
│  │  + Overlays     │  │  + Charts       │  │  + Person Management        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ HTTP/REST + MJPEG Stream
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND LAYER                                   │
│                           (FastAPI + Python 3.11+)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         API ROUTES                                   │   │
│  │  /api/stream/*  │  /api/events/*  │  /api/persons/*  │  /api/stats/* │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ML PIPELINE                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Person Det.  │─►│ Segmentation │─►│ PPE Detection│              │   │
│  │  │ (YOLOv8)     │  │ (SAM3/SAM2)  │  │ (YOLOv11)    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │         │                                    │                       │   │
│  │         ▼                                    ▼                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Tracking     │◄─│ Face Recog.  │◄─│ Association  │              │   │
│  │  │ (DeepSORT)   │  │ (InsightFace)│  │ (Mask-based) │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │         │                                                            │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Temporal     │─►│ Persistence  │─►│ Deduplication│              │   │
│  │  │ Filter (EMA) │  │ Manager      │  │ Manager      │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌───────────────────────────┐  ┌───────────────────────────────────────┐  │
│  │  SQLite/PostgreSQL        │  │  File Storage                         │  │
│  │  - ComplianceEvents       │  │  - Videos (input/processed)           │  │
│  │  - Persons                │  │  - Snapshots                          │  │
│  │  - Face Embeddings        │  │  - Model Weights                      │  │
│  └───────────────────────────┘  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ML Pipeline Architecture

### Detection Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          FRAME PROCESSING PIPELINE                          │
└────────────────────────────────────────────────────────────────────────────┘

Input Frame (BGR)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: PERSON DETECTION                                                 │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ YOLOv8-medium (COCO pretrained)                                    │   │
│ │ - Native tracking enabled                                           │   │
│ │ - Output: [{box, track_id, confidence}, ...]                       │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PERSON SEGMENTATION                                              │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ Priority: SAM3 > SAM2 > Box-based                                  │   │
│ │                                                                     │   │
│ │ SAM3 (facebook/sam3):                                              │   │
│ │   - Streaming video inference                                       │   │
│ │   - Auto session management                                         │   │
│ │   - process_frame(frame, persons) -> {track_id: mask}              │   │
│ │                                                                     │   │
│ │ SAM2 (sam2.1_hiera_base_plus) - Fallback:                          │   │
│ │   - Video propagation every N frames                                │   │
│ │   - Manual track management                                         │   │
│ │   - init_video_tracking() + propagate_masks()                      │   │
│ │                                                                     │   │
│ │ Box-based - Fallback if segmentation fails                         │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: PPE & VIOLATION DETECTION                                        │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ YOLOv11 (custom trained on Safety Lab dataset)                     │   │
│ │                                                                     │   │
│ │ Multi-Scale Detection (if enabled):                                │   │
│ │   Scale 1.0x ──► YOLO ──► Detections A                            │   │
│ │   Scale 1.5x ──► YOLO ──► Detections B (rescaled)                 │   │
│ │   Scale 2.0x ──► YOLO ──► Detections C (rescaled)                 │   │
│ │         │                                                          │   │
│ │         └────────► NMS Merge (IoU 0.5) ────► Final Detections     │   │
│ │                                                                     │   │
│ │ Output Classes:                                                    │   │
│ │   PPE: Googles, Mask, Lab Coat, Gloves, Head Mask                 │   │
│ │   Violations: No googles, No Mask, No Lab coat, No Gloves, etc.   │   │
│ │   Actions: Drinking, Eating                                        │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: PPE-PERSON ASSOCIATION                                           │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ Association Strategy:                                              │   │
│ │                                                                     │   │
│ │ IF both person_mask AND ppe_mask available:                        │   │
│ │     containment = calculate_mask_containment(ppe_mask, person_mask)│   │
│ │ ELSE:                                                              │   │
│ │     containment = calculate_box_containment(ppe_box, person_box)   │   │
│ │                                                                     │   │
│ │ For violations (often small boxes):                                │   │
│ │     associate IF:                                                  │   │
│ │       - center_inside(viol_box, person_box) OR                     │   │
│ │       - containment >= 0.1 OR                                      │   │
│ │       - IoU >= 0.1                                                 │   │
│ │                                                                     │   │
│ │ Output per person:                                                 │   │
│ │   - detected_ppe: ["safety goggles", "lab coat"]                   │   │
│ │   - missing_ppe: ["face mask"]                                     │   │
│ │   - action_violations: ["Drinking"]                                │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: TRACKING & IDENTIFICATION                                        │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ DeepSORT Tracker:                                                  │   │
│ │   - Kalman filter for motion prediction                            │   │
│ │   - Hungarian algorithm for assignment                             │   │
│ │   - Handles occlusions and re-entry                                │   │
│ │                                                                     │   │
│ │ PersonGallery (Re-identification):                                 │   │
│ │   - Stores visual features per person                              │   │
│ │   - Re-identifies across track deletions                           │   │
│ │   - Limited to REID_MAX_GALLERY_SIZE features                      │   │
│ │                                                                     │   │
│ │ InsightFace (ArcFace):                                             │   │
│ │   - 512-dim face embeddings                                        │   │
│ │   - Cosine similarity matching (threshold: 0.6)                    │   │
│ │   - Persistent identity across sessions                            │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: TEMPORAL FILTERING & CONFIDENCE FUSION                           │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ Temporal Buffer: 3 frames                                          │   │
│ │                                                                     │   │
│ │ Confidence Fusion (EMA - Exponential Moving Average):              │   │
│ │   conf_t = α × detection_conf + (1-α) × conf_{t-1}                │   │
│ │   where α = 0.7 (TEMPORAL_EMA_ALPHA)                              │   │
│ │                                                                     │   │
│ │ Alternative strategies:                                            │   │
│ │   - "mean": average across buffer                                  │   │
│ │   - "max": maximum across buffer                                   │   │
│ │                                                                     │   │
│ │ Threshold: 0.4 (TEMPORAL_CONFIDENCE_THRESHOLD)                     │   │
│ │ Min frames for violation: 2 (TEMPORAL_VIOLATION_MIN_FRAMES)       │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: EVENT PERSISTENCE & DEDUPLICATION                                │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │ Deduplication Logic:                                               │   │
│ │   - Track active violations per (person_id, ppe_type)             │   │
│ │   - Create event on violation START                                │   │
│ │   - Update duration while ONGOING                                  │   │
│ │   - Close event on violation END                                   │   │
│ │   - Create new event if violation type CHANGES                     │   │
│ │                                                                     │   │
│ │ Persistence:                                                       │   │
│ │   - Get or create Person record                                    │   │
│ │   - Store face embedding if available                              │   │
│ │   - Create ComplianceEvent with all metadata                       │   │
│ │   - Optional: capture violation snapshot                           │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
Output: Annotated Frame + Events Persisted
```

## Component Details

### Frontend (Next.js 16 + TypeScript)

```
frontend/src/
├── app/
│   └── page.tsx              # Main dashboard with tabs
├── components/
│   ├── video-player.tsx      # Video upload & MJPEG display
│   ├── events-table.tsx      # Compliance events with filters
│   ├── persons-table.tsx     # Person management & name editing
│   ├── stats-card.tsx        # Statistics cards
│   └── charts.tsx            # Analytics charts
└── lib/
    └── api.ts                # API client (fetch wrapper)
```

### Backend (FastAPI + Python 3.11+)

```
backend/app/
├── api/routes/
│   ├── events.py             # Event CRUD endpoints
│   ├── persons.py            # Person management
│   ├── stats.py              # Statistics endpoints
│   └── stream.py             # Video processing & live feed
├── core/
│   ├── config.py             # Pydantic Settings
│   └── database.py           # SQLAlchemy async setup
├── ml/
│   ├── pipeline.py           # Main orchestration
│   ├── detector_factory.py   # Detector selection
│   ├── hybrid_detector.py    # YOLOv8 + YOLOv11 + SAM3/SAM2
│   ├── yolov11_detector.py   # PPE detection + multi-scale
│   ├── person_detector.py    # Person detection with tracking
│   ├── sam3_segmenter.py     # SAM3 streaming video
│   ├── sam2_segmenter.py     # SAM2 video propagation
│   ├── tracker.py            # DeepSORT implementation
│   ├── person_gallery.py     # Re-identification gallery
│   ├── face_recognition.py   # InsightFace wrapper
│   ├── temporal_filter.py    # EMA confidence fusion
│   └── mask_utils.py         # Containment calculations
├── models/
│   ├── event.py              # ComplianceEvent model
│   └── person.py             # Person model
└── services/
    ├── persistence.py        # Event persistence logic
    ├── event_service.py      # Event CRUD operations
    ├── person_service.py     # Person management
    └── deduplication.py      # Violation deduplication
```

## Database Schema

### ComplianceEvent

```python
class ComplianceEvent:
    id: str                    # UUID primary key
    person_id: str             # FK to Person
    track_id: int              # DeepSORT track ID
    timestamp: datetime        # Detection time
    video_source: str          # Video/webcam identifier
    frame_number: int          # Frame index
    detected_ppe: List[str]    # PPE items detected (JSON)
    missing_ppe: List[str]     # Missing required PPE (JSON)
    action_violations: List[str]  # Behavioral violations (JSON)
    is_violation: bool         # True if any violation
    detection_confidence: Dict # Confidence scores (JSON)
    snapshot_path: str         # Path to snapshot image
    start_frame: int           # Violation start frame
    end_frame: int             # Violation end frame
    end_timestamp: datetime    # Violation end time
    duration_frames: int       # Total duration
    is_ongoing: bool           # Still active
```

### Person

```python
class Person:
    id: str                    # UUID primary key
    name: str                  # Display name (editable)
    face_embedding: bytes      # Serialized 512-dim vector
    thumbnail: bytes           # Face thumbnail image
    first_seen: datetime       # First detection time
    last_seen: datetime        # Last detection time
    total_events: int          # Total event count
    violation_count: int       # Violation event count
    compliance_rate: float     # Compliance percentage
```

## Configuration Reference

### Segmentation (SAM3/SAM2)

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_SAM3` | `true` | Enable SAM3 streaming video segmentation |
| `SAM3_MODEL` | `facebook/sam3` | HuggingFace model name |
| `USE_SAM2` | `true` | Enable SAM2 as fallback |
| `SAM2_MODEL_TYPE` | `sam2.1_hiera_base_plus` | SAM2 model variant |
| `SAM2_PROPAGATE_INTERVAL` | `2` | Propagate masks every N frames |
| `SAM2_SEGMENT_PPE` | `true` | Also segment PPE items |

### Multi-Scale Detection

| Setting | Default | Description |
|---------|---------|-------------|
| `MULTI_SCALE_ENABLED` | `true` | Enable multi-scale inference |
| `MULTI_SCALE_FACTORS` | `[1.0, 1.5, 2.0]` | Scale factors to use |
| `MULTI_SCALE_NMS_THRESHOLD` | `0.5` | NMS IoU for merging |

### Temporal Filtering

| Setting | Default | Description |
|---------|---------|-------------|
| `TEMPORAL_BUFFER_SIZE` | `3` | Frames in temporal buffer |
| `TEMPORAL_FUSION_STRATEGY` | `ema` | Fusion method (ema/mean/max) |
| `TEMPORAL_EMA_ALPHA` | `0.7` | EMA weight for recent frame |
| `TEMPORAL_CONFIDENCE_THRESHOLD` | `0.4` | Threshold after fusion |
| `TEMPORAL_VIOLATION_MIN_FRAMES` | `2` | Min frames for violation |

### Detection Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.5` | General detection threshold |
| `VIOLATION_CONFIDENCE_THRESHOLD` | `0.3` | Violation detection threshold |
| `FACE_RECOGNITION_THRESHOLD` | `0.6` | Face matching threshold |
| `MASK_CONTAINMENT_THRESHOLD` | `0.5` | PPE-person association |
| `MASK_DENSITY_THRESHOLD` | `0.1` | Minimum mask density |

## API Endpoints

### Statistics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats/summary` | GET | Dashboard statistics |
| `/api/stats/timeline` | GET | Violation timeline |
| `/api/stats/by-ppe` | GET | Violations by PPE type |

### Events

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/events` | GET | Paginated events list |
| `/api/events/{id}` | GET | Single event details |
| `/api/events/recent/violations` | GET | Recent violations |

### Persons

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/persons` | GET | Paginated persons list |
| `/api/persons/{id}` | GET | Single person details |
| `/api/persons/{id}` | PATCH | Update person (name) |
| `/api/persons/top/violators` | GET | Top violators |

### Video Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream/upload` | POST | Upload video file |
| `/api/stream/process` | POST | Start processing job |
| `/api/stream/jobs` | GET | List all jobs |
| `/api/stream/jobs/{id}` | GET | Job status |
| `/api/stream/processed/{id}` | GET | Get processed video |
| `/api/stream/processed/{id}/stream` | GET | MJPEG stream |
| `/api/stream/live/feed` | GET | Live webcam MJPEG |

## Performance Considerations

1. **Model Singletons**: All ML models are loaded once via `get_*()` functions
2. **Frame Sampling**: Process at `FRAME_SAMPLE_RATE` FPS (default: 10)
3. **SAM3 Streaming**: More efficient than SAM2 for video processing
4. **Multi-Scale Trade-off**: Better accuracy vs. 3x inference time
5. **Temporal Filtering**: Reduces false positives and event count
6. **Event Deduplication**: Prevents database bloat
7. **Async Processing**: Background tasks for video processing
8. **MJPEG Streaming**: Efficient for live feed display

## Security Considerations

- CORS configured for frontend origin only
- Input validation via Pydantic models
- SQL injection prevention via SQLAlchemy ORM
- File upload validation (video formats only)
- No authentication (demo/hackathon scope)

## Future Enhancements

1. **Multi-Camera Support**: Track persons across cameras
2. **Real-time Alerts**: WebSocket push notifications
3. **Export Functionality**: CSV/PDF reports
4. **Authentication**: JWT or API key auth
5. **Industrial Domain**: Add helmet/shoes detection
6. **Edge Deployment**: TensorRT optimization
