"""API routes for PPE compliance system."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import json

from database.connection import get_db
from database.models import ComplianceEvent, Person, VideoSession
from schemas.schemas import (
    ComplianceEventResponse,
    PersonResponse,
    ComplianceStatistics,
    VideoUploadResponse,
    VideoSessionResponse,
    DetectionLogFilter
)
from services.video_service import VideoService
from services.websocket_manager import WebSocketManager
from loguru import logger

api_router = APIRouter()
websocket_manager = WebSocketManager()


# ============================================================================
# Video Upload & Processing
# ============================================================================

@api_router.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload video file for processing."""
    try:
        video_service = VideoService()
        result = await video_service.save_uploaded_video(file, db)
        logger.info(f"Video uploaded: {result['filename']}, session_id: {result['session_id']}")
        return result
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/process-video/{session_id}")
async def process_video(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Start processing uploaded video."""
    try:
        video_service = VideoService()
        await video_service.process_video(session_id, db, websocket_manager)
        return {"message": "Video processing started", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/video-sessions", response_model=List[VideoSessionResponse])
async def get_video_sessions(
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """Get list of video processing sessions."""
    sessions = db.query(VideoSession).order_by(
        VideoSession.upload_time.desc()
    ).limit(limit).offset(offset).all()
    return sessions


@api_router.get("/video-sessions/{session_id}", response_model=VideoSessionResponse)
async def get_video_session(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Get specific video session details."""
    session = db.query(VideoSession).filter(VideoSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Video session not found")
    return session


# ============================================================================
# Compliance Events & Detection Logs
# ============================================================================

@api_router.get("/detections", response_model=List[ComplianceEventResponse])
async def get_detections(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    person_id: Optional[UUID] = None,
    compliance_status: Optional[bool] = None,
    camera_source: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """Get compliance detection logs with filtering."""
    query = db.query(ComplianceEvent)
    
    # Apply filters
    if start_date:
        query = query.filter(ComplianceEvent.timestamp >= start_date)
    if end_date:
        query = query.filter(ComplianceEvent.timestamp <= end_date)
    if person_id:
        query = query.filter(ComplianceEvent.person_id == person_id)
    if compliance_status is not None:
        query = query.filter(ComplianceEvent.compliance_status == compliance_status)
    if camera_source:
        query = query.filter(ComplianceEvent.camera_source == camera_source)
    
    # Order and paginate
    events = query.order_by(
        ComplianceEvent.timestamp.desc()
    ).limit(limit).offset(offset).all()
    
    return events


@api_router.get("/detections/{event_id}", response_model=ComplianceEventResponse)
async def get_detection(
    event_id: UUID,
    db: Session = Depends(get_db)
):
    """Get specific detection event."""
    event = db.query(ComplianceEvent).filter(ComplianceEvent.event_id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Detection event not found")
    return event


# ============================================================================
# Person Management
# ============================================================================

@api_router.get("/persons", response_model=List[PersonResponse])
async def get_persons(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """Get list of tracked persons."""
    persons = db.query(Person).order_by(
        Person.last_seen.desc()
    ).limit(limit).offset(offset).all()
    return persons


@api_router.get("/persons/{person_id}", response_model=PersonResponse)
async def get_person(
    person_id: UUID,
    db: Session = Depends(get_db)
):
    """Get specific person details."""
    person = db.query(Person).filter(Person.person_id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@api_router.get("/persons/{person_id}/events", response_model=List[ComplianceEventResponse])
async def get_person_events(
    person_id: UUID,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """Get all compliance events for a specific person."""
    events = db.query(ComplianceEvent).filter(
        ComplianceEvent.person_id == person_id
    ).order_by(
        ComplianceEvent.timestamp.desc()
    ).limit(limit).offset(offset).all()
    return events


# ============================================================================
# Statistics & Analytics
# ============================================================================

@api_router.get("/statistics", response_model=ComplianceStatistics)
async def get_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get compliance statistics."""
    from sqlalchemy import func
    
    # Default to last 24 hours if no date range specified
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=1)
    if not end_date:
        end_date = datetime.utcnow()
    
    # Query events in date range
    query = db.query(ComplianceEvent).filter(
        ComplianceEvent.timestamp >= start_date,
        ComplianceEvent.timestamp <= end_date
    )
    
    total_events = query.count()
    total_violations = query.filter(ComplianceEvent.compliance_status == False).count()
    total_compliances = query.filter(ComplianceEvent.compliance_status == True).count()
    
    # Count unique persons
    unique_persons = db.query(func.count(func.distinct(ComplianceEvent.person_id))).filter(
        ComplianceEvent.timestamp >= start_date,
        ComplianceEvent.timestamp <= end_date
    ).scalar()
    
    # Count specific PPE violations
    events_with_violations = query.filter(ComplianceEvent.compliance_status == False).all()
    
    helmet_violations = sum(1 for e in events_with_violations if "helmet" in e.missing_ppe)
    shoes_violations = sum(1 for e in events_with_violations if "safety_shoes" in e.missing_ppe)
    goggles_violations = sum(1 for e in events_with_violations if "goggles" in e.missing_ppe)
    mask_violations = sum(1 for e in events_with_violations if "mask" in e.missing_ppe)
    
    compliance_rate = (total_compliances / total_events * 100) if total_events > 0 else 0.0
    
    return ComplianceStatistics(
        total_events=total_events,
        total_violations=total_violations,
        total_compliances=total_compliances,
        compliance_rate=round(compliance_rate, 2),
        unique_persons=unique_persons or 0,
        helmet_violations=helmet_violations,
        shoes_violations=shoes_violations,
        goggles_violations=goggles_violations,
        mask_violations=mask_violations,
        last_updated=datetime.utcnow()
    )


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@api_router.websocket("/ws/video-feed")
async def websocket_video_feed(websocket: WebSocket):
    """WebSocket endpoint for real-time video feed and detection updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive messages from client
            data = await websocket.receive_text()
            # Handle client messages if needed (e.g., control commands)
            logger.debug(f"Received from client: {data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


@api_router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Could handle dashboard-specific commands
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
