"""
Main Detection Pipeline

Orchestrates detection, YOLOv8 native tracking, and temporal filtering.
Uses YOLOv8 native tracking for consistent person track_ids.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from uuid import uuid4

from .detector_factory import get_detector
from .temporal_filter import get_temporal_filter
from .mask_utils import (
    draw_person_with_ppe,
    draw_frame_info,
    get_color,
    COLORS,
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Main pipeline orchestrating all detection and tracking.

    Flow:
    1. Sample frame at target FPS
    2. Detect persons (with YOLOv8 native tracking) and PPE using configured detector
    3. Associate PPE with persons
    4. Apply temporal filtering (with confidence fusion if available)
    5. Generate compliance events
    """

    def __init__(self):
        self.detector = get_detector()
        self.temporal_filter = get_temporal_filter()

        # Frame sampling
        self.target_fps = settings.FRAME_SAMPLE_RATE
        self.frame_count = 0

        # Video state tracking (for hybrid detector)
        self.current_video_source: Optional[str] = None

        # Visualization settings
        self.show_masks = getattr(settings, "SHOW_MASKS", True)
        self.mask_alpha = getattr(settings, "MASK_ALPHA", 0.4)

        # Confidence-based filtering settings
        self.use_confidence_fusion = (
            getattr(settings, "TEMPORAL_FUSION_STRATEGY", "ema") != "none"
        )

    def initialize(self):
        """Initialize all ML models."""
        self.detector.initialize()

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

        # 1. Detect persons (with YOLOv8 native tracking) and PPE
        detections = self.detector.detect(frame)
        persons = detections.get("persons", [])
        ppe_detections = detections.get("ppe_detections", {})
        violation_detections = detections.get("violation_detections", {})
        action_violations = detections.get("action_violations", [])

        # Log detection results for debugging
        total_violations = sum(len(dets) for dets in violation_detections.values())
        total_ppe = sum(len(dets) for dets in ppe_detections.values())
        if total_violations > 0 or total_ppe > 0:
            logger.info(
                f"Pipeline: Frame {self.frame_count} - "
                f"{len(persons)} persons, {total_ppe} PPE items, "
                f"{total_violations} violations detected"
            )
            if total_violations > 0:
                logger.info(f"Violation types: {list(violation_detections.keys())}")
        elif self.frame_count % 30 == 0:  # Log every 30 frames if no detections
            logger.warning(
                f"Pipeline: Frame {self.frame_count} - "
                f"No PPE or violations detected. "
                f"Persons: {len(persons)}"
            )

        # 2. Associate PPE and violations with persons
        persons = self.detector.associate_ppe_to_persons(
            persons, ppe_detections, violation_detections, action_violations
        )

        # Log association results
        associated_violations = sum(
            1 for p in persons if p.get("missing_ppe") or p.get("is_violation", False)
        )
        if associated_violations > 0:
            logger.info(
                f"Pipeline: {associated_violations} persons with violations after association"
            )
            for person in persons:
                if person.get("missing_ppe") or person.get("is_violation", False):
                    logger.info(
                        f"  Person {person.get('track_id')}: "
                        f"missing_ppe={person.get('missing_ppe', [])}, "
                        f"ppe_detections={len(person.get('ppe_detections', []))}"
                    )

        # 3. Process each person (persons already have track_id from YOLOv8)
        for person in persons:
            track_id = person.get("track_id")

            # Use track_id directly as person_id
            if track_id is not None:
                person_id = f"person_{track_id}"
            else:
                # Fallback if tracking not available
                person_id = f"track_{person.get('id', 0)}"
                track_id = person.get("id", 0)

            person_result = {
                "person_id": person_id,
                "track_id": track_id,
                "box": person.get("box", [0, 0, 0, 0]),
                "mask": person.get("mask"),  # SAM2 mask for visualization
                "detected_ppe": person.get("detected_ppe", []),
                "missing_ppe": person.get("missing_ppe", []),
                "action_violations": person.get("action_violations", []),
                "detection_confidence": person.get("detection_confidence", {}),
                "ppe_detections": person.get("ppe_detections", []),
                "is_violation": person.get("is_violation", False),
            }

            # 4. Apply temporal filtering (with confidence fusion if available)
            detection_confidence = person_result.get("detection_confidence", {})

            if self.use_confidence_fusion and detection_confidence:
                # Use confidence-based temporal filtering with EMA fusion
                filter_result = self.temporal_filter.update_with_confidence(
                    person_id, detection_confidence
                )
                person_result["fused_confidence"] = filter_result.get(
                    "fused_confidence", {}
                )
            else:
                # Fall back to binary temporal filtering
                filter_result = self.temporal_filter.update(
                    person_id, person_result.get("missing_ppe", [])
                )

            person_result["stable_violation"] = filter_result["is_violation"]
            person_result["stable_missing_ppe"] = filter_result["stable_missing_ppe"]

            # Check for action violations (Drinking/Eating) - these are immediate violations
            action_viols = person_result.get("action_violations", [])
            has_action_violation = len(action_viols) > 0

            # 5. Generate event if stable PPE violation OR action violation
            if filter_result["is_violation"] or has_action_violation:
                # Combine missing PPE with action violations for the event
                all_violations = list(filter_result["stable_missing_ppe"])
                for av in action_viols:
                    all_violations.append(f"{av['action']} in lab")

                event = {
                    "id": str(uuid4()),
                    "person_id": person_id,
                    "track_id": track_id,
                    "timestamp": timestamp.isoformat(),
                    "video_source": video_source,
                    "frame_number": self.frame_count,
                    "detected_ppe": person_result.get("detected_ppe", []),
                    "missing_ppe": filter_result["stable_missing_ppe"],
                    "action_violations": [av["action"] for av in action_viols],
                    "is_violation": True,
                    "detection_confidence": person_result.get(
                        "detection_confidence", {}
                    ),
                    "fused_confidence": person_result.get("fused_confidence", {}),
                }
                result["events"].append(event)
                result["violations"].append(
                    {
                        "person_id": person_id,
                        "track_id": track_id,
                        "missing_ppe": filter_result["stable_missing_ppe"],
                        "action_violations": [av["action"] for av in action_viols],
                        "box": person_result.get("box"),
                    }
                )

            result["persons"].append(person_result)
            # Also add to tracks for API compatibility
            result["tracks"].append(
                {
                    "track_id": track_id,
                    "person_id": person_id,
                    "box": person_result.get("box"),
                }
            )

        # 6. Annotate frame (include unassociated violations for visualization)
        result["annotated_frame"] = self._annotate_frame(
            frame, result["persons"], violation_detections, action_violations
        )

        return result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        violation_detections: Optional[Dict[str, List[Dict]]] = None,
        action_violations: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        Draw annotations on frame with masks and boxes.

        Uses mask_utils for drawing when masks are available.
        Also draws unassociated violation boxes for debugging.
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

        # Draw unassociated violation boxes (for debugging)
        # These are violations detected by YOLOv11 but not yet associated with a person
        if violation_detections:
            unassociated_count = 0
            for ppe_type, viol_list in violation_detections.items():
                for viol in viol_list:
                    viol_box = viol.get("box", [0, 0, 0, 0])
                    if viol_box != [0, 0, 0, 0]:
                        x1, y1, x2, y2 = [int(c) for c in viol_box]
                        # Draw red box for violations
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Add label without confidence score for violation classes
                        label = f"{ppe_type}"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )
                        unassociated_count += 1
            if unassociated_count > 0:
                logger.info(
                    f"Drawing {unassociated_count} unassociated violation boxes on frame {self.frame_count}"
                )

        # Draw unassociated action violations
        if action_violations:
            for action in action_violations:
                action_box = action.get("box", [0, 0, 0, 0])
                if action_box != [0, 0, 0, 0]:
                    x1, y1, x2, y2 = [int(c) for c in action_box]
                    # Draw red box for action violations
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add label
                    action_type = action.get("action", "violation")
                    label = f"Action: {action_type}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

        # Draw frame info
        annotated = draw_frame_info(
            annotated,
            self.frame_count,
            len(persons),  # Number of tracked persons
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

    def load_known_persons(self, persons: List):
        """Load known persons from database (legacy support, not used with YOLOv8 tracking)."""
        # Not needed with YOLOv8 native tracking, but kept for API compatibility
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "frame_count": self.frame_count,
            "current_video": self.current_video_source,
        }

    def reset(self):
        """Reset pipeline state."""
        self.temporal_filter.clear_all()
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
