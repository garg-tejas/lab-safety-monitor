"""
Hybrid Detector

Combined YOLO + SAM 2 pipeline for person detection, PPE detection, and segmentation.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .person_detector import get_person_detector, PersonDetector
from .yolov8_detector import get_yolov8_detector, YOLOv8Detector
from .sam2_segmenter import get_sam2_segmenter, SAM2Segmenter
from .mask_utils import calculate_mask_containment, calculate_box_containment
from ..core.config import settings


class HybridDetector:
    """
    Combined YOLO + SAM 2 detection pipeline.

    Detection flow:
    1. Run PersonDetector (YOLOv8-nano) → person boxes
    2. Run PPE Detector (YOLOv8 trained) → PPE boxes
    3. Run SAM 2 segmentation → masks for all boxes
    4. Validate masks (density check)
    5. Associate PPE with persons using mask containment
    6. Return combined results
    """

    def __init__(self):
        self.person_detector: Optional[PersonDetector] = None
        self.ppe_detector: Optional[YOLOv8Detector] = None
        self.segmenter: Optional[SAM2Segmenter] = None

        self._initialized = False
        self._use_sam2 = getattr(settings, "USE_SAM2", True)

        # PPE containment threshold for association
        self._containment_threshold = getattr(
            settings, "MASK_CONTAINMENT_THRESHOLD", 0.5
        )

    def initialize(self) -> None:
        """Initialize all sub-detectors."""
        if self._initialized:
            return

        print("Initializing HybridDetector...")

        # Initialize person detector
        self.person_detector = get_person_detector()
        self.person_detector.initialize()

        # Initialize PPE detector
        self.ppe_detector = get_yolov8_detector()
        self.ppe_detector.initialize()

        # Initialize SAM 2 segmenter (optional)
        if self._use_sam2:
            try:
                self.segmenter = get_sam2_segmenter()
                self.segmenter.initialize()
                print("SAM2 segmentation enabled")
            except Exception as e:
                print(f"Warning: SAM2 not available, running without masks: {e}")
                self.segmenter = None
                self._use_sam2 = False
        else:
            print("SAM2 disabled by configuration")

        self._initialized = True
        print("HybridDetector initialized")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full detection pipeline on a frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            Dict with:
                - persons: List of person detections with boxes and masks
                - ppe_detections: Dict of PPE type -> list of detections
        """
        if not self._initialized:
            self.initialize()

        # 1. Detect persons
        persons = self.person_detector.detect(frame)

        # 2. Detect PPE
        ppe_result = self.ppe_detector.detect(frame)
        raw_ppe_detections = ppe_result.get("ppe_detections", {})

        # Flatten PPE detections for segmentation
        all_ppe = []
        for ppe_type, detections in raw_ppe_detections.items():
            for det in detections:
                all_ppe.append(
                    {
                        "box": det["box"],
                        "label": ppe_type,
                        "score": det["score"],
                    }
                )

        # 3. Run SAM 2 segmentation on all boxes
        if self._use_sam2 and self.segmenter is not None:
            # Collect all boxes and labels
            all_boxes = [p["box"] for p in persons]
            all_labels = ["person"] * len(persons)

            for ppe in all_ppe:
                all_boxes.append(ppe["box"])
                all_labels.append(ppe["label"])

            # Segment all at once (box-prompted segmentation)
            if all_boxes:
                segmentation_results = self.segmenter.segment_boxes(
                    frame, all_boxes, all_labels
                )

                # Assign masks back to detections
                seg_idx = 0
                for i in range(len(persons)):
                    seg = segmentation_results[seg_idx]
                    if seg["valid"]:
                        persons[i]["mask"] = seg["mask"]
                        persons[i]["mask_score"] = seg["score"]
                    seg_idx += 1

                for i in range(len(all_ppe)):
                    seg = segmentation_results[seg_idx]
                    if seg["valid"]:
                        all_ppe[i]["mask"] = seg["mask"]
                        all_ppe[i]["mask_score"] = seg["score"]
                    seg_idx += 1

        # 4. Rebuild ppe_detections dict with masks
        ppe_detections: Dict[str, List[Dict]] = {}
        for ppe in all_ppe:
            ppe_type = ppe["label"]
            if ppe_type not in ppe_detections:
                ppe_detections[ppe_type] = []
            ppe_detections[ppe_type].append(ppe)

        return {
            "persons": persons,
            "ppe_detections": ppe_detections,
            "frame_shape": frame.shape[:2],
        }

    def associate_ppe_to_persons(
        self,
        persons: List[Dict[str, Any]],
        ppe_detections: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Associate PPE items with persons using mask containment (or box overlap as fallback).

        Args:
            persons: List of person detections
            ppe_detections: Dict of PPE type -> list of detections

        Returns:
            Updated persons list with detected_ppe and missing_ppe fields
        """
        # Get required PPE from settings
        required_ppe = getattr(
            settings,
            "REQUIRED_PPE",
            ["safety goggles", "face mask", "lab coat", "safety shoes"],
        )

        for person in persons:
            person_mask = person.get("mask")
            person_box = person.get("box", [0, 0, 0, 0])

            detected_ppe = []
            detection_confidence = {}
            person_ppe_detections = []  # PPE assigned to this person

            for ppe_type, ppe_list in ppe_detections.items():
                for ppe in ppe_list:
                    ppe_mask = ppe.get("mask")
                    ppe_box = ppe.get("box", [0, 0, 0, 0])

                    # Calculate containment
                    if person_mask is not None and ppe_mask is not None:
                        # Use mask-based containment (more accurate)
                        containment = calculate_mask_containment(ppe_mask, person_mask)
                    else:
                        # Fall back to box-based containment
                        containment = calculate_box_containment(ppe_box, person_box)

                    # If PPE is mostly inside person, assign it
                    if containment >= self._containment_threshold:
                        if ppe_type not in detected_ppe:
                            detected_ppe.append(ppe_type)
                            detection_confidence[ppe_type] = ppe.get("score", 0.0)

                        # Add to person's PPE list (for visualization)
                        person_ppe_detections.append(
                            {
                                "label": ppe_type,
                                "box": ppe_box,
                                "mask": ppe_mask,
                                "score": ppe.get("score", 0.0),
                                "containment": containment,
                            }
                        )

            # Determine missing PPE
            missing_ppe = [p for p in required_ppe if p not in detected_ppe]

            # Update person
            person["detected_ppe"] = detected_ppe
            person["missing_ppe"] = missing_ppe
            person["detection_confidence"] = detection_confidence
            person["ppe_detections"] = person_ppe_detections
            person["is_violation"] = len(missing_ppe) > 0

        return persons

    def reset_video_state(self) -> None:
        """Reset video tracking state (call when switching videos)."""
        if self.segmenter is not None:
            self.segmenter.reset_video_state()

    def __repr__(self) -> str:
        return (
            f"HybridDetector(initialized={self._initialized}, "
            f"sam2_enabled={self._use_sam2})"
        )


# Singleton instance
_hybrid_detector: Optional[HybridDetector] = None


def get_hybrid_detector() -> HybridDetector:
    """Get singleton HybridDetector instance."""
    global _hybrid_detector
    if _hybrid_detector is None:
        _hybrid_detector = HybridDetector()
    return _hybrid_detector
