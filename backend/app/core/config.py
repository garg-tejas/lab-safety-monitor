from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from pathlib import Path
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App settings
    APP_NAME: str = "MarketWise Lab Safety"
    DEBUG: bool = True

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./marketwise.db"

    # ML Settings
    SAM3_MODEL: str = "facebook/sam3"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    FACE_RECOGNITION_THRESHOLD: float = 0.6

    # Detector selection: "hybrid", "yolov8", "sam3", or "mock"
    # "hybrid" uses YOLO + SAM2 for best results
    DETECTOR_TYPE: str = Field(default="hybrid")

    # SAM 2 Settings
    SAM2_MODEL_TYPE: str = Field(default="sam2.1_hiera_base_plus")
    SAM2_MODEL_PATH: Optional[Path] = None  # Auto-set in validator
    USE_SAM2: bool = Field(default=True)
    USE_SAM2_VIDEO_PROPAGATION: bool = Field(default=True)

    # Mask Validation
    MASK_DENSITY_THRESHOLD: float = Field(default=0.1)  # Reject if < 10%
    MASK_CONTAINMENT_THRESHOLD: float = Field(
        default=0.5
    )  # PPE must be 50% inside person

    # Visualization
    SHOW_MASKS: bool = Field(default=True)
    MASK_ALPHA: float = Field(default=0.4)  # 40% opacity

    # Mock mode (for development/testing without ML models)
    USE_MOCK_DETECTOR: bool = False
    USE_MOCK_FACE: bool = False

    # Video Processing
    FRAME_SAMPLE_RATE: int = 10  # Target FPS for processing
    TEMPORAL_BUFFER_SIZE: int = 3  # Frames for stable detection
    MAX_TRACK_AGE: int = 30  # Max frames to keep unmatched track
    MIN_TRACK_HITS: int = 3  # Min detections before track is confirmed

    # PPE Detection prompts for SAM 3
    PPE_PROMPTS: List[str] = Field(
        default=[
            "safety goggles",
            "protective helmet",
            "face mask",
            "lab coat",
            "safety shoes",
        ]
    )

    # Required PPE items (violations generated if missing)
    # Laboratory Domain Configuration:
    # - "protective helmet" removed - not required in lab environments
    # - "safety shoes" removed - model not trained on shoes, not required in labs
    # To switch to industrial mode, add "protective helmet" and "safety shoes"
    REQUIRED_PPE: List[str] = Field(
        default=[
            "safety goggles",
            "face mask",
            "lab coat",
        ]
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    WEIGHTS_DIR: Optional[Path] = None
    DATA_DIR: Optional[Path] = None
    VIDEOS_DIR: Optional[Path] = None
    SNAPSHOTS_DIR: Optional[Path] = None
    PROCESSED_DIR: Optional[Path] = None  # Directory for annotated/processed videos
    ENABLE_SNAPSHOT_CAPTURE: bool = True
    YOLOV8_MODEL_PATH: Optional[Path] = None

    @model_validator(mode="after")
    def set_derived_paths(self):
        """Set derived paths after initialization."""
        # Set WEIGHTS_DIR
        if self.WEIGHTS_DIR is None:
            object.__setattr__(self, "WEIGHTS_DIR", self.BASE_DIR / "weights")

        # Set DATA_DIR
        if self.DATA_DIR is None:
            object.__setattr__(self, "DATA_DIR", self.BASE_DIR.parent / "data")

        # Set VIDEOS_DIR
        if self.VIDEOS_DIR is None:
            object.__setattr__(self, "VIDEOS_DIR", self.DATA_DIR / "videos")

        # Set SNAPSHOTS_DIR
        if self.SNAPSHOTS_DIR is None:
            object.__setattr__(self, "SNAPSHOTS_DIR", self.DATA_DIR / "snapshots")

        # Set PROCESSED_DIR for annotated videos
        if self.PROCESSED_DIR is None:
            object.__setattr__(self, "PROCESSED_DIR", self.DATA_DIR / "processed")

        # Set YOLOV8_MODEL_PATH
        if self.YOLOV8_MODEL_PATH is None:
            # Default to ONNX model if available, else PyTorch
            onnx_path = self.WEIGHTS_DIR / "ppe_detector" / "best.onnx"
            pt_path = self.WEIGHTS_DIR / "ppe_detector" / "best.pt"
            if onnx_path.exists():
                object.__setattr__(self, "YOLOV8_MODEL_PATH", onnx_path)
            elif pt_path.exists():
                object.__setattr__(self, "YOLOV8_MODEL_PATH", pt_path)
            else:
                # Default to pt path even if doesn't exist (will be handled by detector)
                object.__setattr__(self, "YOLOV8_MODEL_PATH", pt_path)

        # Set SAM2_MODEL_PATH
        if self.SAM2_MODEL_PATH is None:
            sam2_path = self.WEIGHTS_DIR / "sam2" / "sam2.1_hiera_base_plus.pt"
            object.__setattr__(self, "SAM2_MODEL_PATH", sam2_path)

        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
