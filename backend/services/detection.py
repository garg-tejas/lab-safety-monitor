"""Detection service using YOLOv8 for person and PPE detection."""
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger
import sys

sys.path.append('..')
from config import settings


class DetectionService:
    """Service for detecting persons and PPE equipment."""
    
    def __init__(self):
        """Initialize YOLO models."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load person detection model (pretrained YOLO)
        try:
            from ultralytics import YOLO
            self.person_model = YOLO(settings.YOLO_PERSON_MODEL)
            logger.info("Person detection model loaded")
        except Exception as e:
            logger.warning(f"Could not load person model: {e}. Will use fallback detection.")
            self.person_model = None
        
        # Load PPE detection model (fine-tuned)
        try:
            if Path(settings.YOLO_PPE_MODEL).exists():
                self.ppe_model = YOLO(settings.YOLO_PPE_MODEL)
                logger.info("PPE detection model loaded")
            else:
                logger.warning(f"PPE model not found at {settings.YOLO_PPE_MODEL}")
                self.ppe_model = None
        except Exception as e:
            logger.warning(f"Could not load PPE model: {e}")
            self.ppe_model = None
        
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of person detections with bounding boxes
        """
        if self.person_model is None:
            logger.warning("Person model not available")
            return []
        
        try:
            results = self.person_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person class in COCO dataset
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        },
                        'confidence': confidence,
                        'class': 'person'
                    })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def detect_ppe(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect PPE equipment in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of PPE detections with bounding boxes and classes
        """
        if self.ppe_model is None:
            logger.warning("PPE model not available, using mock detections")
            return self._mock_ppe_detection(frame)
        
        try:
            results = self.ppe_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    detections.append({
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        },
                        'confidence': confidence,
                        'class': class_name
                    })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in PPE detection: {e}")
            return []
    
    def _mock_ppe_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Mock PPE detection for testing when model is not available.
        Randomly generates PPE detections based on image regions.
        """
        height, width = frame.shape[:2]
        mock_detections = []
        
        # Simulate helmet detection in upper region
        if np.random.random() > 0.3:
            mock_detections.append({
                'bbox': {
                    'x1': width * 0.3,
                    'y1': height * 0.1,
                    'x2': width * 0.5,
                    'y2': height * 0.3
                },
                'confidence': 0.85,
                'class': 'helmet'
            })
        
        return mock_detections
    
    def associate_ppe_with_persons(
        self,
        person_detections: List[Dict],
        ppe_detections: List[Dict]
    ) -> List[Dict]:
        """
        Associate PPE equipment with detected persons based on spatial proximity.
        
        Args:
            person_detections: List of person detections
            ppe_detections: List of PPE detections
            
        Returns:
            List of persons with associated PPE
        """
        persons_with_ppe = []
        
        for person in person_detections:
            person_bbox = person['bbox']
            person_center_x = (person_bbox['x1'] + person_bbox['x2']) / 2
            person_center_y = (person_bbox['y1'] + person_bbox['y2']) / 2
            
            # Initialize PPE status
            ppe_status = {
                'helmet': False,
                'safety_shoes': False,
                'goggles': False,
                'mask': False
            }
            
            # Find PPE items near this person
            for ppe in ppe_detections:
                ppe_bbox = ppe['bbox']
                ppe_center_x = (ppe_bbox['x1'] + ppe_bbox['x2']) / 2
                ppe_center_y = (ppe_bbox['y1'] + ppe_bbox['y2']) / 2
                
                # Check if PPE is within person's region using spatial logic
                if self._is_ppe_associated_with_person(
                    person_bbox, ppe_bbox, ppe['class']
                ):
                    ppe_class = ppe['class'].lower()
                    if ppe_class in ppe_status:
                        ppe_status[ppe_class] = True
            
            # Determine missing PPE
            missing_ppe = [k for k, v in ppe_status.items() if not v]
            is_compliant = len(missing_ppe) == 0
            
            persons_with_ppe.append({
                'bbox': person_bbox,
                'confidence': person['confidence'],
                'ppe_detected': ppe_status,
                'missing_ppe': missing_ppe,
                'is_compliant': is_compliant
            })
        
        return persons_with_ppe
    
    def _is_ppe_associated_with_person(
        self,
        person_bbox: Dict,
        ppe_bbox: Dict,
        ppe_class: str
    ) -> bool:
        """
        Determine if PPE belongs to a person based on spatial relationship.
        
        Uses vertical position logic:
        - Helmet: above or overlapping upper part of person
        - Goggles/Mask: middle-upper region
        - Safety shoes: lower part of person
        """
        # Calculate overlap using IoU
        iou = self._calculate_iou(person_bbox, ppe_bbox)
        
        # If significant overlap, PPE belongs to person
        if iou > 0.1:
            return True
        
        # Check vertical position for specific PPE types
        person_height = person_bbox['y2'] - person_bbox['y1']
        ppe_center_y = (ppe_bbox['y1'] + ppe_bbox['y2']) / 2
        
        if ppe_class == 'helmet':
            # Helmet should be in upper 40% of person bbox
            if ppe_center_y < person_bbox['y1'] + person_height * 0.4:
                return self._horizontal_overlap(person_bbox, ppe_bbox)
        
        elif ppe_class in ['goggles', 'mask']:
            # Face gear in middle-upper region (20-60%)
            if person_bbox['y1'] + person_height * 0.2 < ppe_center_y < person_bbox['y1'] + person_height * 0.6:
                return self._horizontal_overlap(person_bbox, ppe_bbox)
        
        elif ppe_class == 'safety_shoes':
            # Shoes in lower 30% of person bbox
            if ppe_center_y > person_bbox['y1'] + person_height * 0.7:
                return self._horizontal_overlap(person_bbox, ppe_bbox)
        
        return False
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _horizontal_overlap(self, person_bbox: Dict, ppe_bbox: Dict) -> bool:
        """Check if PPE has horizontal overlap with person."""
        return not (ppe_bbox['x2'] < person_bbox['x1'] or ppe_bbox['x1'] > person_bbox['x2'])
