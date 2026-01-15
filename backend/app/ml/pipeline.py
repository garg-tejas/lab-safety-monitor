"""
Main Detection Pipeline

Orchestrates detection, face recognition, DeepSORT tracking, and temporal filtering.
Now supports YOLO + SAM2 hybrid detection with mask visualization.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Generator
from datetime import datetime
from uuid import uuid4

from .detector_factory import get_detector
from .face_recognition import get_face_recognizer, FaceRecognizer
from .temporal_filter import get_temporal_filter
from .tracker import get_tracker, DeepSORTTracker
from .mask_utils import (
    draw_person_with_ppe,
    draw_frame_info,
    get_color,
    COLORS,
)
from ..core.config import settings


class DetectionPipeline:
    """
    Main pipeline orchestrating all detection and tracking.

    Flow:
    1. Sample frame at target FPS
    2. Detect persons and PPE using configured detector
    3. Run DeepSORT tracker for consistent person tracking
    4. Detect faces and extract embeddings
    5. Link face identities to tracks
    6. Apply temporal filtering
    7. Generate compliance events
    """

    def __init__(self):
        self.detector = get_detector()
        self.face_recognizer = get_face_recognizer()
        self.temporal_filter = get_temporal_filter()
        self.tracker = get_tracker()

        # Known persons: id -> embedding
        self.known_persons: Dict[str, np.ndarray] = {}

        # Frame sampling
        self.target_fps = settings.FRAME_SAMPLE_RATE
        self.frame_count = 0

        # Video state tracking (for hybrid detector)
        self.current_video_source: Optional[str] = None

        # Visualization settings
        self.show_masks = getattr(settings, "SHOW_MASKS", True)
        self.mask_alpha = getattr(settings, "MASK_ALPHA", 0.4)

    def initialize(self):
        """Initialize all ML models."""
        self.detector.initialize()
        self.face_recognizer.initialize()

    def process_frame(
        self, frame: np.ndarray, video_source: str = "video"
    ) -> Dict[str, Any]:
        """
        Process a single frame through the entire pipeline.

        Args:
            frame: BGR numpy array from OpenCV
            video_source: Source identifier for logging

        Returns:
            Dict with detection results, violations, and events
        """
        # Reset video state if source changed
        if video_source != self.current_video_source:
            self.current_video_source = video_source
            if hasattr(self.detector, "reset_video_state"):
                self.detector.reset_video_state()
            self.tracker.reset()
            self.temporal_filter.clear_all()
            self.frame_count = 0

        self.frame_count += 1
        timestamp = datetime.now()

        result = {
            "frame_number": self.frame_count,
            "timestamp": timestamp.isoformat(),
            "persons": [],
            "violations": [],
            "events": [],
            "tracks": [],
            "annotated_frame": None,
        }

        # 1. Detect persons and PPE
        detections = self.detector.detect(frame)
        persons = detections.get("persons", [])
        ppe_detections = detections.get("ppe_detections", {})

        # 2. Associate PPE with persons
        persons = self.detector.associate_ppe_to_persons(persons, ppe_detections)

        # 3. Detect faces and extract embeddings for appearance features
        faces = self.face_recognizer.detect_faces(frame)

        # 4. Prepare detections for tracker (add appearance features)
        tracker_detections = []
        for person in persons:
            det = {
                "box": person.get("box", [0, 0, 0, 0]),
                "appearance_feature": None,
                "original_person": person,
            }

            # Try to get face embedding as appearance feature
            person_box = person.get("box", [0, 0, 0, 0])
            for face in faces:
                face_box = face.get("box", [0, 0, 0, 0])
                if self._boxes_overlap(person_box, face_box):
                    if face.get("embedding") is not None:
                        det["appearance_feature"] = face["embedding"]
                        det["face"] = face
                    break

            tracker_detections.append(det)

        # 5. Update DeepSORT tracker
        tracked_objects = self.tracker.update(tracker_detections)
        result["tracks"] = tracked_objects

        # 6. Build person results with consistent track IDs
        track_to_person_map = self._build_track_person_map(
            tracked_objects, tracker_detections, faces
        )

        for track_info in tracked_objects:
            track_id = track_info["track_id"]
            person_data = track_to_person_map.get(track_id, {})

            # Get person_id (from face recognition) or use track-based ID
            person_id, face_embedding = self._get_person_id_for_track(track_info, faces)

            # Link person_id to track for future frames
            if person_id and not person_id.startswith("track_"):
                self.tracker.link_person_id(track_id, person_id)
            elif track_info.get("person_id"):
                # Use previously linked person_id
                person_id = track_info["person_id"]
                face_embedding = None
            else:
                person_id = f"track_{track_id}"
                face_embedding = None

            person_result = {
                "person_id": person_id,
                "track_id": track_id,
                "track_state": track_info.get("state", "unknown"),
                "box": track_info.get("box", [0, 0, 0, 0]),
                "mask": person_data.get("mask"),
                "detected_ppe": person_data.get("detected_ppe", []),
                "missing_ppe": person_data.get("missing_ppe", []),
                "detection_confidence": person_data.get("detection_confidence", {}),
                "ppe_detections": person_data.get("ppe_detections", []),
                "face_embedding": face_embedding,
            }

            # 7. Apply temporal filtering (only for confirmed tracks)
            if track_info.get("state") == "confirmed":
                filter_result = self.temporal_filter.update(
                    person_id, person_result.get("missing_ppe", [])
                )

                person_result["stable_violation"] = filter_result["is_violation"]
                person_result["stable_missing_ppe"] = filter_result[
                    "stable_missing_ppe"
                ]

                # 8. Generate event if stable violation
                if filter_result["is_violation"]:
                    event = {
                        "id": str(uuid4()),
                        "person_id": person_id,
                        "track_id": track_id,
                        "timestamp": timestamp.isoformat(),
                        "video_source": video_source,
                        "frame_number": self.frame_count,
                        "detected_ppe": person_result.get("detected_ppe", []),
                        "missing_ppe": filter_result["stable_missing_ppe"],
                        "is_violation": True,
                        "face_embedding": person_result.get("face_embedding"),
                        "detection_confidence": person_result.get(
                            "detection_confidence", {}
                        ),
                    }
                    result["events"].append(event)
                    result["violations"].append(
                        {
                            "person_id": person_id,
                            "track_id": track_id,
                            "missing_ppe": filter_result["stable_missing_ppe"],
                            "box": person_result.get("box"),
                        }
                    )
            else:
                person_result["stable_violation"] = False
                person_result["stable_missing_ppe"] = []

            result["persons"].append(person_result)

        # 9. Annotate frame with masks
        result["annotated_frame"] = self._annotate_frame(frame, result["persons"])

        return result

    def _build_track_person_map(
        self,
        tracked_objects: List[Dict],
        tracker_detections: List[Dict],
        faces: List[Dict],
    ) -> Dict[int, Dict]:
        """
        Map track IDs to original person detection data.
        Uses IOU matching between track boxes and detection boxes.
        """
        track_to_person = {}

        for track in tracked_objects:
            track_box = track.get("box", [0, 0, 0, 0])
            best_iou = 0.0
            best_person = None

            for det in tracker_detections:
                det_box = det.get("box", [0, 0, 0, 0])
                iou = self._calculate_iou(track_box, det_box)

                if iou > best_iou:
                    best_iou = iou
                    best_person = det.get("original_person", {})

            if best_person and best_iou > 0.3:
                track_to_person[track["track_id"]] = {
                    "detected_ppe": best_person.get("detected_ppe", []),
                    "missing_ppe": best_person.get("missing_ppe", []),
                    "detection_confidence": best_person.get("detection_confidence", {}),
                    "mask": best_person.get("mask"),
                    "ppe_detections": best_person.get("ppe_detections", []),
                }

        return track_to_person

    def _get_person_id_for_track(
        self, track_info: Dict, faces: List[Dict]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Identify person for a track using face recognition.
        Returns (person_id, embedding).
        """
        track_box = track_info.get("box", [0, 0, 0, 0])

        # Find face that overlaps with track box
        for face in faces:
            face_box = face.get("box", [0, 0, 0, 0])
            if self._boxes_overlap(track_box, face_box):
                embedding = face.get("embedding")
                if embedding is not None:
                    # Try to match with known persons
                    known_list = list(self.known_persons.items())
                    match = self.face_recognizer.find_matching_person(
                        embedding, known_list
                    )

                    if match:
                        person_id, _similarity = match
                        return person_id, embedding
                    else:
                        # New person - register them
                        person_id = f"person_{len(self.known_persons) + 1}"
                        self.known_persons[person_id] = embedding
                        return person_id, embedding

        return None, None

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two boxes overlap."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return x2 > x1 and y2 > y1

    def _annotate_frame(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """
        Draw annotations on frame with masks and boxes.

        Uses mask_utils for drawing when masks are available.
        """
        annotated = frame.copy()

        num_violations = sum(1 for p in persons if p.get("stable_violation", False))

        for person in persons:
            # Get PPE detections for this person
            ppe_detections = person.get("ppe_detections", [])

            # Draw person with PPE using mask_utils
            annotated = draw_person_with_ppe(
                annotated,
                person,
                ppe_detections,
                show_masks=self.show_masks,
                mask_alpha=self.mask_alpha,
            )

        # Draw frame info
        annotated = draw_frame_info(
            annotated,
            self.frame_count,
            len(self.tracker.get_confirmed_tracks()),
            num_violations,
        )

        return annotated

    def process_video(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Process a video file frame by frame.

        Yields results for each processed frame (sampled at target FPS).
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / self.target_fps))
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at target FPS
                if frame_idx % frame_skip == 0:
                    result = self.process_frame(frame, video_source=video_path)
                    result["video_frame_idx"] = frame_idx
                    yield result

                frame_idx += 1
        finally:
            cap.release()

    def load_known_persons(self, persons: List[Tuple[str, bytes]]):
        """Load known persons from database."""
        for person_id, embedding_bytes in persons:
            embedding = FaceRecognizer.deserialize_embedding(embedding_bytes)
            self.known_persons[person_id] = embedding

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "frame_count": self.frame_count,
            "known_persons": len(self.known_persons),
            "active_tracks": len(self.tracker.tracks),
            "confirmed_tracks": len(self.tracker.get_confirmed_tracks()),
            "current_video": self.current_video_source,
        }

    def reset(self):
        """Reset pipeline state."""
        self.temporal_filter.clear_all()
        self.tracker.reset()
        self.frame_count = 0
        self.current_video_source = None

        # Reset detector video state if applicable
        if hasattr(self.detector, "reset_video_state"):
            self.detector.reset_video_state()


# Singleton instance
_pipeline = None


def get_pipeline() -> DetectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline
