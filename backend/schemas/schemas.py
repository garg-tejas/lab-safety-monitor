"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID


class PPEDetection(BaseModel):
    """PPE detection result."""
    helmet: bool
    safety_shoes: bool
    goggles: bool
    mask: bool


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PersonDetectionResult(BaseModel):
    """Detection result for a single person."""
    person_id: str
    bbox: BoundingBox
    ppe_detected: PPEDetection
    missing_ppe: List[str]
    is_compliant: bool
    face_detected: bool
    confidence: float


class FrameDetectionResult(BaseModel):
    """Detection result for entire frame."""
    frame_number: int
    timestamp: float
    persons_detected: List[PersonDetectionResult]
    total_persons: int
    compliant_count: int
    violation_count: int


class ComplianceEventCreate(BaseModel):
    """Schema for creating compliance event."""
    person_id: UUID
    camera_source: str
    detected_ppe: Dict[str, bool]
    missing_ppe: List[str]
    compliance_status: bool
    person_bbox: Optional[Dict[str, float]] = None
    detection_confidence: Optional[float] = None
    frame_snapshot_path: Optional[str] = None


class ComplianceEventResponse(BaseModel):
    """Schema for compliance event response."""
    model_config = ConfigDict(from_attributes=True)
    
    event_id: UUID
    person_id: UUID
    timestamp: datetime
    camera_source: str
    detected_ppe: Dict[str, bool]
    missing_ppe: List[str]
    compliance_status: bool
    detection_confidence: Optional[float]
    frame_snapshot_path: Optional[str]


class PersonResponse(BaseModel):
    """Schema for person response."""
    model_config = ConfigDict(from_attributes=True)
    
    person_id: UUID
    first_seen: datetime
    last_seen: datetime
    total_violations: int
    total_compliances: int


class ComplianceStatistics(BaseModel):
    """Compliance statistics."""
    total_events: int
    total_violations: int
    total_compliances: int
    compliance_rate: float
    unique_persons: int
    helmet_violations: int
    shoes_violations: int
    goggles_violations: int
    mask_violations: int
    last_updated: datetime


class VideoUploadResponse(BaseModel):
    """Response for video upload."""
    session_id: UUID
    filename: str
    file_size: int
    upload_time: datetime
    status: str


class VideoSessionResponse(BaseModel):
    """Video session details."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: UUID
    video_name: str
    upload_time: datetime
    processing_status: str
    duration_seconds: Optional[float]
    total_frames: Optional[int]
    fps: Optional[float]
    frames_processed: int
    total_detections: int
    total_violations: int


class DetectionLogFilter(BaseModel):
    """Filter parameters for detection logs."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    person_id: Optional[UUID] = None
    compliance_status: Optional[bool] = None
    camera_source: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "detection", "statistics", "error", "info"
    data: Dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
