"""Compliance engine for evaluating PPE compliance and managing person data."""
import numpy as np
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger
import sys

sys.path.append('..')
from database.models import Person, ComplianceEvent
from services.face_recognition import FaceRecognitionService


class ComplianceEngine:
    """Engine for evaluating PPE compliance and managing person tracking."""
    
    def __init__(self):
        """Initialize compliance engine."""
        self.face_service = FaceRecognitionService()
        self.person_cache = {}  # In-memory cache for active persons
    
    def process_tracked_persons(
        self,
        tracked_persons: List[Dict],
        faces_detected: List[Dict],
        camera_source: str,
        db: Session
    ) -> List[Dict]:
        """
        Process tracked persons, associate with faces, and evaluate compliance.
        
        Args:
            tracked_persons: List of tracked persons with PPE info
            faces_detected: List of detected faces with embeddings
            camera_source: Video source identifier
            db: Database session
            
        Returns:
            List of compliance results
        """
        compliance_results = []
        
        for tracked_person in tracked_persons:
            # Associate face with tracked person
            face_info = self._associate_face_with_person(
                tracked_person, faces_detected
            )
            
            # Find or create person in database
            person_id = self._get_or_create_person(
                face_info, db
            )
            
            # Evaluate compliance
            ppe_detected = tracked_person.get('ppe_detected', {
                'helmet': False,
                'safety_shoes': False,
                'goggles': False,
                'mask': False
            })
            
            missing_ppe = tracked_person.get('missing_ppe', [])
            is_compliant = tracked_person.get('is_compliant', False)
            
            # Create compliance event
            event = self._create_compliance_event(
                person_id=person_id,
                camera_source=camera_source,
                ppe_detected=ppe_detected,
                missing_ppe=missing_ppe,
                is_compliant=is_compliant,
                bbox=tracked_person.get('bbox'),
                confidence=tracked_person.get('confidence'),
                db=db
            )
            
            compliance_results.append({
                'event_id': str(event.event_id),
                'person_id': str(person_id),
                'track_id': tracked_person.get('track_id'),
                'ppe_detected': ppe_detected,
                'missing_ppe': missing_ppe,
                'is_compliant': is_compliant,
                'bbox': tracked_person.get('bbox'),
                'face_detected': face_info is not None
            })
        
        return compliance_results
    
    def _associate_face_with_person(
        self,
        tracked_person: Dict,
        faces_detected: List[Dict]
    ) -> Optional[Dict]:
        """
        Find face that belongs to tracked person based on bbox overlap.
        
        Args:
            tracked_person: Tracked person with bbox
            faces_detected: List of detected faces
            
        Returns:
            Face info dict or None
        """
        if not faces_detected:
            return None
        
        person_bbox = tracked_person.get('bbox')
        if not person_bbox:
            return None
        
        # Find face with best overlap
        best_face = None
        best_overlap = 0.0
        
        for face in faces_detected:
            face_bbox = face.get('bbox')
            if not face_bbox:
                continue
            
            # Check if face is within person bbox
            overlap = self._calculate_bbox_overlap(person_bbox, face_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_face = face
        
        # Require minimum 30% overlap
        return best_face if best_overlap > 0.3 else None
    
    def _calculate_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        
        return intersection / area2 if area2 > 0 else 0.0
    
    def _get_or_create_person(
        self,
        face_info: Optional[Dict],
        db: Session
    ) -> UUID:
        """
        Find existing person or create new one based on face embedding.
        
        Args:
            face_info: Face information with embedding
            db: Database session
            
        Returns:
            Person UUID
        """
        if not face_info or face_info.get('embedding') is None:
            # No face detected, create anonymous person
            return self._create_anonymous_person(db)
        
        face_embedding = np.array(face_info['embedding'])
        
        # Query recent persons from database
        recent_persons = db.query(Person).order_by(
            Person.last_seen.desc()
        ).limit(100).all()
        
        # Try to match with existing person
        for person in recent_persons:
            if person.face_embedding is None:
                continue
            
            db_embedding = np.array(person.face_embedding)
            similarity, is_match = self.face_service.compare_faces(
                face_embedding, db_embedding
            )
            
            if is_match:
                # Update last seen time
                person.last_seen = datetime.utcnow()
                db.commit()
                logger.debug(f"Matched existing person: {person.person_id}")
                return person.person_id
        
        # No match found, create new person
        return self._create_new_person(face_embedding, db)
    
    def _create_new_person(
        self,
        face_embedding: np.ndarray,
        db: Session
    ) -> UUID:
        """Create new person in database."""
        person = Person(
            person_id=uuid4(),
            face_embedding=face_embedding.tolist(),
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            total_violations=0,
            total_compliances=0
        )
        db.add(person)
        db.commit()
        logger.info(f"Created new person: {person.person_id}")
        return person.person_id
    
    def _create_anonymous_person(self, db: Session) -> UUID:
        """Create anonymous person (no face detected)."""
        person = Person(
            person_id=uuid4(),
            face_embedding=None,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            total_violations=0,
            total_compliances=0,
            notes="Anonymous - no face detected"
        )
        db.add(person)
        db.commit()
        return person.person_id
    
    def _create_compliance_event(
        self,
        person_id: UUID,
        camera_source: str,
        ppe_detected: Dict[str, bool],
        missing_ppe: List[str],
        is_compliant: bool,
        bbox: Optional[Dict],
        confidence: Optional[float],
        db: Session
    ) -> ComplianceEvent:
        """Create compliance event in database."""
        event = ComplianceEvent(
            event_id=uuid4(),
            person_id=person_id,
            timestamp=datetime.utcnow(),
            camera_source=camera_source,
            detected_ppe=ppe_detected,
            missing_ppe=missing_ppe,
            compliance_status=is_compliant,
            person_bbox=bbox,
            detection_confidence=confidence,
            frame_snapshot_path=None  # Will be set if snapshot is saved
        )
        
        db.add(event)
        
        # Update person statistics
        person = db.query(Person).filter(Person.person_id == person_id).first()
        if person:
            if is_compliant:
                person.total_compliances += 1
            else:
                person.total_violations += 1
            person.last_seen = datetime.utcnow()
        
        db.commit()
        
        return event
