"""Database models for PPE compliance system."""
from sqlalchemy import Column, String, Integer, Boolean, Float, DateTime, JSON, LargeBinary, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Person(Base):
    """Person entity tracked by the system."""
    __tablename__ = "persons"
    
    person_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    face_embedding = Column(ARRAY(Float), nullable=True)  # 512-dim vector
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    total_violations = Column(Integer, default=0, nullable=False)
    total_compliances = Column(Integer, default=0, nullable=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    compliance_events = relationship("ComplianceEvent", back_populates="person")
    
    def __repr__(self):
        return f"<Person(id={self.person_id}, violations={self.total_violations})>"


class ComplianceEvent(Base):
    """Individual compliance/violation event."""
    __tablename__ = "compliance_events"
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.person_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    camera_source = Column(String(100), nullable=False)
    
    # Detection results
    detected_ppe = Column(JSON, nullable=False)  # {"helmet": true, "shoes": false, ...}
    missing_ppe = Column(ARRAY(String), nullable=False)  # ["goggles", "mask"]
    compliance_status = Column(Boolean, nullable=False)  # True if all PPE present
    
    # Bounding box coordinates
    person_bbox = Column(JSON, nullable=True)  # {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    
    # Confidence scores
    detection_confidence = Column(Float, nullable=True)
    
    # Snapshot
    frame_snapshot_path = Column(String(500), nullable=True)
    
    # Relationships
    person = relationship("Person", back_populates="compliance_events")
    
    def __repr__(self):
        return f"<ComplianceEvent(id={self.event_id}, compliant={self.compliance_status})>"


class VideoSession(Base):
    """Video processing session."""
    __tablename__ = "video_sessions"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_name = Column(String(255), nullable=False)
    video_path = Column(String(500), nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status = Column(String(50), default="pending", nullable=False)  # pending, processing, completed, failed
    
    # Video metadata
    duration_seconds = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    resolution = Column(String(50), nullable=True)  # "1920x1080"
    
    # Processing stats
    frames_processed = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    
    processing_started = Column(DateTime, nullable=True)
    processing_completed = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<VideoSession(id={self.session_id}, status={self.processing_status})>"


class SystemLog(Base):
    """System logs and audit trail."""
    __tablename__ = "system_logs"
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True)
    details = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<SystemLog(level={self.level}, message={self.message[:50]})>"
