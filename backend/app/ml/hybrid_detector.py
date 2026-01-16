"""
Hybrid Detector

Combined YOLOv8 (person tracking) + YOLOv11 (PPE detection) pipeline.
Uses YOLOv8 native tracking for consistent person track_ids.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

from .person_detector import get_person_detector, PersonDetector
from .yolov11_detector import get_yolov11_detector, YOLOv11Detector
from .mask_utils import calculate_box_containment
from ..core.config import settings

logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Combined YOLOv8 (person tracking) + YOLOv11 (PPE detection) pipeline.

    Detection flow:
    1. Run PersonDetector with tracking (YOLOv8-medium) → person boxes with track_ids
    2. Run PPE Detector (YOLOv11 custom trained) → PPE boxes + violation boxes
    3. Associate PPE with persons using box overlap
    4. Return combined results with track_ids
    """

    def __init__(self):
        self.person_detector: Optional[PersonDetector] = None
        self.ppe_detector: Optional[YOLOv11Detector] = None

        self._initialized = False

        # PPE containment threshold for association (box overlap)
        # Lower threshold (0.3) allows for better association when PPE is partially visible
        self._containment_threshold = getattr(
            settings, "MASK_CONTAINMENT_THRESHOLD", 0.3
        )
        # Even lower threshold for violations (0.1) since violation boxes may be very small
        # We also use center-point and IoU checks for better association
        self._violation_containment_threshold = 0.1

    def initialize(self) -> None:
        """Initialize all sub-detectors."""
        if self._initialized:
            return

        print("Initializing HybridDetector...")

        # Initialize person detector (YOLOv8-medium with tracking)
        self.person_detector = get_person_detector()
        self.person_detector.initialize()

        # Initialize PPE detector (YOLOv11)
        self.ppe_detector = get_yolov11_detector()
        self.ppe_detector.initialize()
        
        # Verify YOLOv11 model is loaded (not in mock mode)
        if self.ppe_detector.model is None:
            logger.error("⚠️ YOLOv11 model is None - PPE and violation detection will NOT work!")
            logger.error("Check that model file exists at: {}".format(
                settings.YOLOV11_MODEL_PATH if settings.YOLOV11_MODEL_PATH else "Not set"
            ))
        else:
            logger.info(f"✓ YOLOv11 model loaded and ready (Type: {self.ppe_detector.model_type})")

        self._initialized = True
        print("HybridDetector initialized")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full detection pipeline on a frame with tracking.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            Dict with:
                - persons: List of person detections with boxes and track_ids
                - ppe_detections: Dict of PPE type -> list of detections
                - violation_detections: Dict of PPE type -> list of "No X" detections
                - action_violations: List of action violations (Drinking/Eating)
        """
        if not self._initialized:
            self.initialize()

        # 1. Detect persons with YOLOv8 native tracking
        persons = self.person_detector.detect_with_tracking(frame)

        # 2. Detect PPE and violations (YOLOv11 returns both)
        if self.ppe_detector.model is None:
            logger.error("=" * 80)
            logger.error("⚠️ YOLOv11 model is None - cannot detect PPE or violations!")
            logger.error("Check model loading logs above for errors.")
            logger.error("=" * 80)
        
        # Call YOLOv11 detector
        logger.debug(f"Calling YOLOv11 detect() on frame shape: {frame.shape}")
        ppe_result = self.ppe_detector.detect(frame)
        ppe_detections = ppe_result.get("ppe_detections", {})
        violation_detections = ppe_result.get("violation_detections", {})
        action_violations = ppe_result.get("action_violations", [])
        
        # Log detection results for debugging
        total_violations = sum(len(dets) for dets in violation_detections.values())
        total_ppe = sum(len(dets) for dets in ppe_detections.values())
        
        # Always log (even if 0) so we can see what's happening
        logger.info(
            f"HybridDetector: YOLOv11 returned {total_ppe} PPE items, "
            f"{total_violations} violations, {len(action_violations)} action violations"
        )
        
        if total_violations == 0 and total_ppe == 0 and self.ppe_detector.model is not None:
            logger.warning(
                f"⚠️ YOLOv11 model is loaded but returned 0 detections. "
                f"Check confidence thresholds and model output."
            )

        return {
            "persons": persons,
            "ppe_detections": ppe_detections,
            "violation_detections": violation_detections,
            "action_violations": action_violations,
            "frame_shape": frame.shape[:2],
        }

    def associate_ppe_to_persons(
        self,
        persons: List[Dict[str, Any]],
        ppe_detections: Dict[str, List[Dict[str, Any]]],
        violation_detections: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        action_violations: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Associate PPE items and violations with persons using box overlap.

        The model uses a "No X" class pattern for violation detection:
        - "Googles" = Person wearing goggles (compliant)
        - "No googles" = Person NOT wearing goggles (violation!)

        Args:
            persons: List of person detections (with track_ids)
            ppe_detections: Dict of PPE type -> list of positive detections
            violation_detections: Dict of PPE type -> list of "No X" detections
            action_violations: List of action violations (Drinking/Eating)

        Returns:
            Updated persons list with detected_ppe, missing_ppe, and action_violations fields
        """
        # Get configuration from settings
        class_map = getattr(settings, "PPE_CLASS_MAP", {})
        violation_classes = getattr(settings, "VIOLATION_CLASSES", [])
        action_violation_classes = getattr(settings, "ACTION_VIOLATION_CLASSES", [])
        required_ppe = settings.REQUIRED_PPE

        violation_detections = violation_detections or {}
        action_violations = action_violations or []

        for person in persons:
            person_box = person.get("box", [0, 0, 0, 0])

            detected_ppe = []
            missing_ppe = []
            person_action_violations = []
            detection_confidence = {}
            person_ppe_detections = []

            # Check for positive PPE detections (e.g., "Googles", "Mask", "Lab Coat")
            for ppe_class, ppe_list in ppe_detections.items():
                # Skip violation classes (they start with "No")
                if ppe_class.startswith("No "):
                    continue

                for ppe in ppe_list:
                    ppe_box = ppe.get("box", [0, 0, 0, 0])

                    # Calculate box-based containment
                    containment = calculate_box_containment(ppe_box, person_box)

                    # If PPE overlaps significantly with person, assign it
                    if containment >= self._containment_threshold:
                        # Map class name to required PPE name
                        ppe_name = class_map.get(ppe_class, ppe_class)

                        if ppe_name not in detected_ppe:
                            detected_ppe.append(ppe_name)
                            detection_confidence[ppe_name] = ppe.get("score", 0.0)

                        # Add to person's PPE list (for visualization)
                        person_ppe_detections.append(
                            {
                                "label": ppe_class,
                                "display_name": ppe_name,
                                "box": ppe_box,
                                "score": ppe.get("score", 0.0),
                                "containment": round(containment, 2),
                                "is_violation": False,
                            }
                        )

            # Check for "No X" violations (e.g., "No googles", "No Mask")
            # These indicate the person is NOT wearing that PPE
            for viol_class, viol_list in violation_detections.items():
                for viol in viol_list:
                    viol_box = viol.get("box", [0, 0, 0, 0])

                    # For small violation boxes, use multiple association methods:
                    # 1. Check if violation box center is inside person box
                    viol_center_x = (viol_box[0] + viol_box[2]) / 2
                    viol_center_y = (viol_box[1] + viol_box[3]) / 2
                    center_inside = (
                        person_box[0] <= viol_center_x <= person_box[2]
                        and person_box[1] <= viol_center_y <= person_box[3]
                    )
                    
                    # 2. Calculate box-based containment (violation box inside person box)
                    containment = calculate_box_containment(viol_box, person_box)
                    # Also check reverse containment (person box inside violation box)
                    reverse_containment = calculate_box_containment(person_box, viol_box)
                    # Use maximum for better matching
                    max_containment = max(containment, reverse_containment)
                    
                    # 3. Calculate IoU for better matching of overlapping boxes
                    iou = self._calculate_iou(viol_box, person_box)

                    # Associate if center is inside OR containment/IoU is above threshold
                    # For violations, be very permissive since boxes are often small
                    should_associate = (
                        center_inside 
                        or max_containment >= self._violation_containment_threshold
                        or iou >= 0.1  # Very low IoU threshold for small boxes
                    )

                    if should_associate:
                        # viol_class is already the PPE type name from VIOLATION_CLASS_MAPPING
                        # (e.g., "safety goggles", "face mask", "lab coat")
                        # No need to do replace("No ", "") since it's already the correct PPE type
                        required_name = viol_class  # viol_class is already the PPE type

                        # Check if this is actually a required PPE item
                        if required_name not in required_ppe:
                            logger.warning(
                                f"Violation detected but not in required_ppe: {required_name}. "
                                f"Required PPE: {required_ppe}. "
                                f"This violation will not trigger alerts."
                            )
                        
                        if (
                            required_name in required_ppe
                            and required_name not in missing_ppe
                        ):
                            missing_ppe.append(required_name)
                            detection_confidence[f"no_{required_name}"] = viol.get(
                                "score", 0.0
                            )
                            logger.info(
                                f"✓ Violation added to missing_ppe: person={person.get('track_id')}, "
                                f"ppe_type={required_name}, score={viol.get('score', 0.0):.3f}"
                            )

                        # Always add to person's PPE list as a violation (for visualization)
                        # even if not in required_ppe
                        person_ppe_detections.append(
                            {
                                "label": viol_class,
                                "display_name": required_name,
                                "box": viol_box,
                                "score": viol.get("score", 0.0),
                                "containment": round(max_containment, 2),
                                "iou": round(iou, 2),
                                "is_violation": True,
                            }
                        )
                        logger.info(
                            f"✓ Violation associated: person={person.get('track_id')}, "
                            f"ppe_type={required_name}, containment={max_containment:.2f}, "
                            f"iou={iou:.2f}, center_inside={center_inside}, "
                            f"score={viol.get('score', 0.0):.3f}, "
                            f"in_required_ppe={required_name in required_ppe}"
                        )
                    else:
                        # Log when violations are NOT associated for debugging
                        logger.warning(
                            f"✗ Violation NOT associated: person={person.get('track_id')}, "
                            f"ppe_type={viol_class}, containment={max_containment:.2f}, "
                            f"iou={iou:.2f}, center_inside={center_inside}, "
                            f"threshold={self._violation_containment_threshold}, "
                            f"viol_box={[round(b, 1) for b in viol_box]}, "
                            f"person_box={[round(b, 1) for b in person_box]}"
                        )

            # Check for action violations (Drinking/Eating)
            for action in action_violations:
                action_box = action.get("box", [0, 0, 0, 0])
                action_class = action.get("class", action.get("action", ""))

                containment = calculate_box_containment(action_box, person_box)

                # Action violations should be associated if they overlap with person
                if containment >= self._containment_threshold:
                    person_action_violations.append(
                        {
                            "action": action_class,
                            "score": action.get("score", 0.0),
                            "containment": round(containment, 2),
                        }
                    )

            # Update person
            person["detected_ppe"] = detected_ppe
            person["missing_ppe"] = missing_ppe
            person["action_violations"] = person_action_violations
            person["detection_confidence"] = detection_confidence
            person["ppe_detections"] = person_ppe_detections
            person["is_violation"] = (
                len(missing_ppe) > 0 or len(person_action_violations) > 0
            )

        return persons

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def reset_video_state(self) -> None:
        """Reset video tracking state (call when switching videos)."""
        # YOLOv8 tracking is handled internally, no reset needed
        pass

    def __repr__(self) -> str:
        return f"HybridDetector(initialized={self._initialized})"


# Singleton instance
_hybrid_detector: Optional[HybridDetector] = None


def get_hybrid_detector() -> HybridDetector:
    """Get singleton HybridDetector instance."""
    global _hybrid_detector
    if _hybrid_detector is None:
        _hybrid_detector = HybridDetector()
    return _hybrid_detector
