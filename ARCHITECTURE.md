# MarketWise Lab Safety - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                          │
│   Live Video  │  Dashboard  │  Events  │  Persons                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ REST / MJPEG
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI)                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                        ML PIPELINE                              │ │
│  │  Person Detection → Segmentation → PPE Detection → Association │ │
│  │     (YOLOv8)        (SAM3/SAM2)     (YOLOv11)                  │ │
│  │                           ↓                                     │ │
│  │      Tracking ← Face Recognition ← Temporal Filter ← Events    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│   SQLite/PostgreSQL  │  Model Weights  │  Videos  │  Snapshots      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

### Stage 1: Person Detection (YOLOv8)

```
Input Frame
     ↓
YOLOv8-medium (COCO pretrained)
     ↓
Persons: [{box, track_id, confidence}, ...]
```

- Native tracking for consistent IDs
- Auto-downloads on first run

### Stage 2: Segmentation (SAM3 / SAM2)

```
Persons + Frame
     ↓
┌─────────────────┐
│ SAM3 (Preferred)│ → process_frame(frame, persons)
│ via ModelScope  │    Single call, streaming video
└─────────────────┘
     ↓ fallback
┌─────────────────┐
│ SAM2 (Fallback) │ → init + propagate
│ Meta's sam2 pkg │    Manual state management
└─────────────────┘
     ↓
Masks: {track_id: mask, ...}
```

**SAM3 vs SAM2:**

| Feature | SAM3 | SAM2 |
|---------|------|------|
| API | `process_frame()` | `init()` + `propagate()` |
| Session | Auto-managed | Manual |
| Source | ModelScope | Meta GitHub |
| Config | `USE_SAM3=true` | `USE_SAM2=true` |

### Stage 3: PPE Detection (YOLOv11)

```
Frame
  ↓
┌──────────────────────────────┐
│ Multi-Scale Detection        │
│  Scale 1.0x → YOLO → Dets A │
│  Scale 1.5x → YOLO → Dets B │
│  Scale 2.0x → YOLO → Dets C │
│       ↓                      │
│  NMS Merge (IoU 0.5)        │
└──────────────────────────────┘
  ↓
PPE Classes: Googles, Mask, Lab Coat, Gloves, Head Mask
Violations:  No googles, No Mask, No Lab coat, No Gloves, No Head Mask
Actions:     Drinking, Eating
```

### Stage 4: Association

```
Persons + PPE/Violations
         ↓
┌─────────────────────────────────────┐
│ IF mask available:                  │
│   containment = mask_containment()  │
│ ELSE:                               │
│   containment = box_containment()   │
│                                     │
│ For violations (small boxes):       │
│   center_inside OR                  │
│   containment >= 0.1 OR             │
│   IoU >= 0.1                        │
└─────────────────────────────────────┘
         ↓
Person: {detected_ppe, missing_ppe, is_violation}
```

### Stage 5: Temporal Filtering

```
Detection Confidence
         ↓
EMA Fusion: conf_t = α × conf + (1-α) × conf_{t-1}
            α = 0.7 (TEMPORAL_EMA_ALPHA)
         ↓
Threshold: 0.4 (TEMPORAL_CONFIDENCE_THRESHOLD)
Min frames: 2 (TEMPORAL_VIOLATION_MIN_FRAMES)
         ↓
Stable Violations
```

### Stage 6: Event Persistence

```
Violation Detected
         ↓
Deduplication: Track (person_id, ppe_type)
  - Create on START
  - Update duration while ONGOING
  - Close on END
         ↓
ComplianceEvent → Database
```

---

## File Structure

```
backend/app/
├── api/routes/
│   ├── events.py       # Event CRUD
│   ├── persons.py      # Person management
│   ├── stats.py        # Statistics
│   └── stream.py       # Video processing
├── core/
│   ├── config.py       # Settings (Pydantic)
│   └── database.py     # SQLAlchemy async
├── ml/
│   ├── pipeline.py         # Main orchestration
│   ├── hybrid_detector.py  # YOLOv8 + YOLOv11 + SAM
│   ├── yolov11_detector.py # PPE detection
│   ├── person_detector.py  # Person detection
│   ├── sam3_segmenter.py   # SAM3 (ModelScope)
│   ├── sam2_segmenter.py   # SAM2 (fallback)
│   ├── temporal_filter.py  # EMA fusion
│   └── mask_utils.py       # Containment calcs
├── models/
│   ├── event.py        # ComplianceEvent
│   └── person.py       # Person
└── services/
    ├── persistence.py  # Event persistence
    └── deduplication.py
```

---

## Database Schema

### ComplianceEvent

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| person_id | UUID | FK to Person |
| track_id | int | DeepSORT ID |
| timestamp | datetime | Detection time |
| detected_ppe | JSON | PPE items found |
| missing_ppe | JSON | Missing required PPE |
| is_violation | bool | Has violation |
| is_ongoing | bool | Still active |

### Person

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| name | str | Display name |
| face_embedding | bytes | 512-dim vector |
| violation_count | int | Total violations |

---

## Configuration

### Segmentation

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_SAM3` | `true` | Enable SAM3 |
| `SAM3_MODEL_PATH` | `weights/sam3/sam3.pt` | SAM3 weights |
| `USE_SAM2` | `true` | Enable SAM2 fallback |
| `SAM2_MODEL_TYPE` | `sam2.1_hiera_base_plus` | SAM2 variant |

### Detection

| Setting | Default | Description |
|---------|---------|-------------|
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.5` | PPE threshold |
| `VIOLATION_CONFIDENCE_THRESHOLD` | `0.3` | Violation threshold |
| `MULTI_SCALE_ENABLED` | `true` | Multi-scale detection |
| `MULTI_SCALE_FACTORS` | `[1.0, 1.5, 2.0]` | Scale factors |

### Temporal

| Setting | Default | Description |
|---------|---------|-------------|
| `TEMPORAL_FUSION_STRATEGY` | `ema` | Fusion method |
| `TEMPORAL_EMA_ALPHA` | `0.7` | EMA weight |
| `TEMPORAL_CONFIDENCE_THRESHOLD` | `0.4` | Post-fusion threshold |

---

## Model Weights

```
backend/weights/
├── person_detector/
│   └── yolov8m.pt           # Auto-downloads
├── ppe_detector/
│   └── best.pt              # YOLOv11 trained model
├── sam3/
│   └── sam3.pt              # ModelScope download
└── sam2/
    └── sam2.1_hiera_base_plus.pt
```

### SAM3 Download (ModelScope)

```bash
pip install modelscope
modelscope download --model facebook/sam3 sam3/sam3.pt --local_dir backend/weights/sam3
```

---

## API Endpoints

### Video Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream/upload` | POST | Upload video |
| `/api/stream/process` | POST | Start processing |
| `/api/stream/live/feed` | GET | Live MJPEG |

### Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats/summary` | GET | Dashboard stats |
| `/api/events` | GET | Event list |
| `/api/persons` | GET | Person list |

---

## Performance

1. **Singletons**: Models loaded once via `get_*()` functions
2. **Frame Sampling**: Process at `FRAME_SAMPLE_RATE` FPS
3. **SAM3 Streaming**: More efficient than SAM2
4. **Multi-Scale Trade-off**: 3x inference for better small object detection
5. **Temporal Filter**: Reduces false positives
6. **Event Deduplication**: Prevents database bloat
