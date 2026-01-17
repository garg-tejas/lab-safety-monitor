"""Application configuration settings."""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "PPE Safety Compliance System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ppe_compliance"
    DB_ECHO: bool = False
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Model Paths
    YOLO_PERSON_MODEL: str = "models/weights/yolov8n.pt"
    YOLO_PPE_MODEL: str = "models/weights/yolov8_ppe.pt"
    FACENET_MODEL: str = "vggface2"
    
    # Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    MAX_PERSONS_PER_FRAME: int = 10
    FRAME_SKIP: int = 2
    
    # Video Processing
    MAX_UPLOAD_SIZE_MB: int = 500
    SUPPORTED_FORMATS: str = "mp4,avi,mov,mkv"
    FPS_TARGET: int = 30
    
    # Face Recognition
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    FACE_EMBEDDING_DIM: int = 512
    
    # Storage
    UPLOAD_DIR: str = "../data/uploads"
    SNAPSHOT_DIR: str = "../data/snapshots"
    VIDEO_DIR: str = "../data/videos"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # PPE Classes
    PPE_CLASSES: List[str] = ["helmet", "safety_shoes", "goggles", "mask"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        Path(settings.UPLOAD_DIR),
        Path(settings.SNAPSHOT_DIR),
        Path(settings.VIDEO_DIR),
        Path("models/weights"),
        Path("logs"),
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()
