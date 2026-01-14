"""Video processing service."""
import cv2
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import UploadFile
from loguru import logger
import asyncio
import sys

sys.path.append('..')
from config import settings
from database.models import VideoSession
from services.detection import DetectionService
from services.face_recognition import FaceRecognitionService
from services.tracking import TrackingService
from services.compliance import ComplianceEngine


class VideoService:
    """Service for video upload, processing, and analysis."""
    
    def __init__(self):
        """Initialize video service."""
        self.detection_service = DetectionService()
        self.face_service = FaceRecognitionService()
        self.tracking_service = TrackingService()
        self.compliance_engine = ComplianceEngine()
        
        # Ensure upload directory exists
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_video(
        self,
        file: UploadFile,
        db: Session
    ) -> Dict[str, Any]:
        """
        Save uploaded video file and create session.
        
        Args:
            file: Uploaded video file
            db: Database session
            
        Returns:
            Dict with upload info
        """
        try:
            # Generate unique filename
            session_id = uuid4()
            file_extension = Path(file.filename).suffix
            filename = f"{session_id}{file_extension}"
            file_path = Path(settings.UPLOAD_DIR) / filename
            
            # Save file
            contents = await file.read()
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            file_size = len(contents)
            
            # Get video metadata
            metadata = self._get_video_metadata(str(file_path))
            
            # Create video session in database
            session = VideoSession(
                session_id=session_id,
                video_name=file.filename,
                video_path=str(file_path),
                upload_time=datetime.utcnow(),
                processing_status='pending',
                duration_seconds=metadata.get('duration'),
                total_frames=metadata.get('total_frames'),
                fps=metadata.get('fps'),
                resolution=metadata.get('resolution')
            )
            
            db.add(session)
            db.commit()
            
            logger.info(f"Video saved: {filename}, size: {file_size} bytes")
            
            return {
                'session_id': session_id,
                'filename': file.filename,
                'file_size': file_size,
                'upload_time': datetime.utcnow(),
                'status': 'pending'
            }
        
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            raise
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'duration': duration
            }
        
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {}
    
    async def process_video(
        self,
        session_id: UUID,
        db: Session,
        websocket_manager=None
    ):
        """
        Process video for PPE compliance detection.
        
        Args:
            session_id: Video session ID
            db: Database session
            websocket_manager: WebSocket manager for real-time updates
        """
        session = db.query(VideoSession).filter(
            VideoSession.session_id == session_id
        ).first()
        
        if not session:
            raise ValueError(f"Video session {session_id} not found")
        
        try:
            # Update status
            session.processing_status = 'processing'
            session.processing_started = datetime.utcnow()
            db.commit()
            
            # Process video
            await self._process_video_frames(
                session, db, websocket_manager
            )
            
            # Update status
            session.processing_status = 'completed'
            session.processing_completed = datetime.utcnow()
            db.commit()
            
            logger.info(f"Video processing completed: {session_id}")
        
        except Exception as e:
            logger.error(f"Error processing video {session_id}: {e}")
            session.processing_status = 'failed'
            session.error_message = str(e)
            db.commit()
            raise
    
    async def _process_video_frames(
        self,
        session: VideoSession,
        db: Session,
        websocket_manager=None
    ):
        """Process individual video frames."""
        cap = cv2.VideoCapture(session.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {session.video_path}")
        
        frame_count = 0
        processed_count = 0
        camera_source = f"video_{session.session_id}"
        
        # Reset tracker for new video
        self.tracking_service.reset()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % settings.FRAME_SKIP != 0:
                continue
            
            # Process frame
            try:
                results = await self._process_single_frame(
                    frame, camera_source, db
                )
                
                processed_count += 1
                session.frames_processed = processed_count
                session.total_detections += len(results)
                session.total_violations += sum(
                    1 for r in results if not r['is_compliant']
                )
                
                # Send WebSocket update
                if websocket_manager and processed_count % 10 == 0:
                    await websocket_manager.broadcast({
                        'type': 'processing_progress',
                        'data': {
                            'session_id': str(session.session_id),
                            'frames_processed': processed_count,
                            'total_frames': session.total_frames,
                            'detections': len(results)
                        }
                    })
                
                # Commit periodically
                if processed_count % 50 == 0:
                    db.commit()
            
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        db.commit()
        
        logger.info(
            f"Processed {processed_count} frames, "
            f"{session.total_detections} detections, "
            f"{session.total_violations} violations"
        )
    
    async def _process_single_frame(
        self,
        frame: np.ndarray,
        camera_source: str,
        db: Session
    ) -> list:
        """Process a single video frame."""
        # Detect persons
        person_detections = self.detection_service.detect_persons(frame)
        
        # Detect PPE
        ppe_detections = self.detection_service.detect_ppe(frame)
        
        # Associate PPE with persons
        persons_with_ppe = self.detection_service.associate_ppe_with_persons(
            person_detections, ppe_detections
        )
        
        # Track persons
        tracked_persons = self.tracking_service.update_tracks(
            persons_with_ppe, frame
        )
        
        # Detect faces
        faces_detected = self.face_service.detect_faces(frame)
        
        # Process compliance
        compliance_results = self.compliance_engine.process_tracked_persons(
            tracked_persons, faces_detected, camera_source, db
        )
        
        return compliance_results
    
    def process_webcam_frame(
        self,
        frame: np.ndarray,
        camera_source: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Process single frame from webcam (synchronous).
        
        Args:
            frame: Video frame
            camera_source: Camera identifier
            db: Database session
            
        Returns:
            Detection results with annotations
        """
        try:
            # Detect persons
            person_detections = self.detection_service.detect_persons(frame)
            
            # Detect PPE
            ppe_detections = self.detection_service.detect_ppe(frame)
            
            # Associate PPE with persons
            persons_with_ppe = self.detection_service.associate_ppe_with_persons(
                person_detections, ppe_detections
            )
            
            # Track persons
            tracked_persons = self.tracking_service.update_tracks(
                persons_with_ppe, frame
            )
            
            # Detect faces
            faces_detected = self.face_service.detect_faces(frame)
            
            # Process compliance
            compliance_results = self.compliance_engine.process_tracked_persons(
                tracked_persons, faces_detected, camera_source, db
            )
            
            # Annotate frame
            annotated_frame = self._annotate_frame(
                frame, tracked_persons, faces_detected
            )
            
            return {
                'frame': annotated_frame,
                'detections': compliance_results,
                'person_count': len(tracked_persons),
                'compliant_count': sum(1 for r in compliance_results if r['is_compliant']),
                'violation_count': sum(1 for r in compliance_results if not r['is_compliant'])
            }
        
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
            return {
                'frame': frame,
                'detections': [],
                'person_count': 0,
                'compliant_count': 0,
                'violation_count': 0
            }
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracked_persons: list,
        faces_detected: list
    ) -> np.ndarray:
        """Add annotations to frame for visualization."""
        annotated = frame.copy()
        
        # Draw person bounding boxes
        for person in tracked_persons:
            bbox = person.get('bbox')
            if not bbox:
                continue
            
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Color based on compliance
            is_compliant = person.get('is_compliant', False)
            color = (0, 255, 0) if is_compliant else (0, 0, 255)  # Green or Red
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID: {person.get('track_id', 'N/A')}"
            if not is_compliant:
                missing = ', '.join(person.get('missing_ppe', []))
                label += f" | Missing: {missing}"
            
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Draw face bounding boxes
        for face in faces_detected:
            bbox = face.get('bbox')
            if not bbox:
                continue
            
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan
        
        return annotated
