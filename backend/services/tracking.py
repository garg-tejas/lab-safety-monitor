"""Person tracking service using DeepSORT."""
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
import sys

sys.path.append('..')
from config import settings


class TrackingService:
    """Service for tracking persons across video frames using DeepSORT."""
    
    def __init__(self):
        """Initialize DeepSORT tracker."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7,
                max_cosine_distance=0.3,
                nn_budget=100,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=True
            )
            logger.info("DeepSORT tracker initialized")
            self.tracker_available = True
            
        except Exception as e:
            logger.warning(f"Could not initialize DeepSORT: {e}. Using simple tracking.")
            self.tracker_available = False
            self.simple_tracks = {}  # Simple ID tracking fallback
            self.next_id = 1
    
    def update_tracks(
        self,
        detections: List[Dict],
        frame: np.ndarray
    ) -> List[Dict]:
        """
        Update person tracks with new detections.
        
        Args:
            detections: List of person detections with bboxes
            frame: Current video frame
            
        Returns:
            List of tracked persons with track IDs
        """
        if self.tracker_available:
            return self._update_tracks_deepsort(detections, frame)
        else:
            return self._update_tracks_simple(detections)
    
    def _update_tracks_deepsort(
        self,
        detections: List[Dict],
        frame: np.ndarray
    ) -> List[Dict]:
        """Update tracks using DeepSORT."""
        try:
            # Convert detections to DeepSORT format
            # Format: ([x1, y1, width, height], confidence, class_name)
            deepsort_detections = []
            for det in detections:
                bbox = det['bbox']
                x1, y1 = bbox['x1'], bbox['y1']
                width = bbox['x2'] - bbox['x1']
                height = bbox['y2'] - bbox['y1']
                
                deepsort_detections.append((
                    [x1, y1, width, height],
                    det.get('confidence', 0.5),
                    'person'
                ))
            
            # Update tracker
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
            
            # Convert tracks back to our format
            tracked_persons = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                
                # Find corresponding detection to get additional info
                detection_info = self._find_matching_detection(
                    ltrb, detections
                )
                
                tracked_person = {
                    'track_id': str(track_id),
                    'bbox': {
                        'x1': float(ltrb[0]),
                        'y1': float(ltrb[1]),
                        'x2': float(ltrb[2]),
                        'y2': float(ltrb[3])
                    },
                    'confidence': detection_info.get('confidence', 0.5) if detection_info else 0.5
                }
                
                # Add PPE information if available
                if detection_info and 'ppe_detected' in detection_info:
                    tracked_person['ppe_detected'] = detection_info['ppe_detected']
                    tracked_person['missing_ppe'] = detection_info['missing_ppe']
                    tracked_person['is_compliant'] = detection_info['is_compliant']
                
                tracked_persons.append(tracked_person)
            
            return tracked_persons
        
        except Exception as e:
            logger.error(f"Error in DeepSORT tracking: {e}")
            return self._update_tracks_simple(detections)
    
    def _update_tracks_simple(self, detections: List[Dict]) -> List[Dict]:
        """
        Simple tracking fallback using IoU matching.
        Not as robust as DeepSORT but works without additional dependencies.
        """
        tracked_persons = []
        
        # Match detections to existing tracks
        unmatched_detections = []
        for det in detections:
            matched = False
            best_iou = 0.3  # Minimum IoU threshold
            best_track_id = None
            
            # Find best matching existing track
            for track_id, track_bbox in self.simple_tracks.items():
                iou = self._calculate_iou(det['bbox'], track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
                    matched = True
            
            if matched:
                # Update existing track
                self.simple_tracks[best_track_id] = det['bbox']
                tracked_person = det.copy()
                tracked_person['track_id'] = str(best_track_id)
                tracked_persons.append(tracked_person)
            else:
                unmatched_detections.append(det)
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_track_id = self.next_id
            self.next_id += 1
            self.simple_tracks[new_track_id] = det['bbox']
            
            tracked_person = det.copy()
            tracked_person['track_id'] = str(new_track_id)
            tracked_persons.append(tracked_person)
        
        # Clean up old tracks (simple timeout)
        if len(self.simple_tracks) > 50:
            # Keep only recent tracks
            active_ids = {tp['track_id'] for tp in tracked_persons}
            self.simple_tracks = {
                k: v for k, v in self.simple_tracks.items()
                if str(k) in active_ids
            }
        
        return tracked_persons
    
    def _find_matching_detection(
        self,
        track_bbox: List[float],
        detections: List[Dict]
    ) -> Optional[Dict]:
        """Find detection that best matches track bounding box."""
        best_iou = 0.3
        best_detection = None
        
        track_bbox_dict = {
            'x1': track_bbox[0],
            'y1': track_bbox[1],
            'x2': track_bbox[2],
            'y2': track_bbox[3]
        }
        
        for det in detections:
            iou = self._calculate_iou(det['bbox'], track_bbox_dict)
            if iou > best_iou:
                best_iou = iou
                best_detection = det
        
        return best_detection
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate IoU between two bounding boxes."""
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
    
    def reset(self):
        """Reset tracker state."""
        if self.tracker_available:
            # Reinitialize tracker
            self.__init__()
        else:
            self.simple_tracks = {}
            self.next_id = 1
