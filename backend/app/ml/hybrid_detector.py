"""
Hybrid Detector

Combined YOLOv8 (person tracking) + YOLOv11 (PPE detection) + SAM3 (segmentation) pipeline.
Uses YOLOv8 native tracking for consistent person track_ids.
SAM3 provides high-quality masks for better PPE-person association with streaming video support.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set

from .person_detector import get_person_detector, PersonDetector
from .yolov11_detector import get_yolov11_detector, YOLOv11Detector
from .mask_utils import calculate_box_containment, calculate_mask_containment
from ..core.config import settings

logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Combined YOLOv8 (person tracking) + YOLOv11 (PPE detection) + SAM3 (segmentation) pipeline.

    Detection flow:
    1. Run PersonDetector with tracking (YOLOv8-medium) -> person boxes with track_ids
    2. Run SAM3 to generate/propagate person masks (streaming video mode)
    3. Run PPE Detector (YOLOv11 custom trained) -> PPE boxes + violation boxes
    4. Generate PPE masks with SAM3 (single-frame mode)
    5. Associate PPE with persons using mask or box overlap
    6. Return combined results with track_ids and masks
    """

    def __init__(self):
        self.person_detector: Optional[PersonDetector] = None
        self.ppe_detector: Optional[YOLOv11Detector] = None
        self.sam3_segmenter = None  # SAM3 for streaming video segmentation
        self.sam2_segmenter = None  # Legacy SAM2 fallback (kept for compatibility)

        self._initialized = False
        self._use_sam3 = getattr(settings, "USE_SAM3", True)
        self._use_sam2 = getattr(settings, "USE_SAM2", True) and not self._use_sam3
        self._use_sam2_video = getattr(settings, "USE_SAM2_VIDEO_PROPAGATION", True)
        self._sam2_propagate_interval = getattr(settings, "SAM2_PROPAGATE_INTERVAL", 2)
        self._segment_ppe = getattr(settings, "SAM2_SEGMENT_PPE", True)

        # Video state tracking
        self._frame_count = 0
        self._video_initialized = False
        self._last_track_ids: Set[int] = set()

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
            logger.error(
                "YOLOv11 model is None - PPE and violation detection will NOT work!"
            )
            logger.error(
                "Check that model file exists at: {}".format(
                    settings.YOLOV11_MODEL_PATH
                    if settings.YOLOV11_MODEL_PATH
                    else "Not set"
                )
            )
        else:
            logger.info(
                f"YOLOv11 model loaded and ready (Type: {self.ppe_detector.model_type})"
            )

        # Initialize SAM3 segmenter (preferred)
        print(f"SAM3 enabled in config: {self._use_sam3}")
        if self._use_sam3:
            try:
                print("Attempting to initialize SAM3 segmenter...")
                from .sam3_segmenter import get_sam3_segmenter

                self.sam3_segmenter = get_sam3_segmenter()
                self.sam3_segmenter.initialize()
                print("SAM3 segmenter initialized with streaming video support")
                logger.info("SAM3 segmenter initialized with streaming video support")
            except Exception as e:
                print(f"SAM3 initialization failed: {e}")
                import traceback

                traceback.print_exc()
                logger.warning(f"SAM3 initialization failed, trying SAM2 fallback: {e}")
                self.sam3_segmenter = None
                self._use_sam3 = False
                # Try SAM2 fallback
                self._use_sam2 = True

        # Initialize SAM2 segmenter as fallback (if SAM3 failed or disabled)
        if not self._use_sam3 and self._use_sam2:
            print(f"SAM2 fallback enabled in config: {self._use_sam2}")
            try:
                print("Attempting to initialize SAM2 segmenter...")
                from .sam2_segmenter import get_sam2_segmenter

                self.sam2_segmenter = get_sam2_segmenter()
                self.sam2_segmenter.initialize()
                print(
                    f"SAM2 segmenter initialized: video_propagation={self._use_sam2_video}"
                )
                logger.info(
                    f"SAM2 segmenter initialized: video_propagation={self._use_sam2_video}"
                )
            except Exception as e:
                print(f"SAM2 initialization failed: {e}")
                import traceback

                traceback.print_exc()
                logger.warning(
                    f"SAM2 initialization failed, falling back to box-based: {e}"
                )
                self.sam2_segmenter = None
                self._use_sam2 = False
        elif not self._use_sam3:
            print("SAM2 is disabled in config")

        self._initialized = True
        segmenter_status = (
            "SAM3"
            if self.sam3_segmenter
            else "SAM2"
            if self.sam2_segmenter
            else "None (box-based)"
        )
        print(f"HybridDetector initialized (Segmenter: {segmenter_status})")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full detection pipeline on a frame with tracking.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            Dict with:
                - persons: List of person detections with boxes, track_ids, and masks
                - ppe_detections: Dict of PPE type -> list of detections
                - violation_detections: Dict of PPE type -> list of "No X" detections
                - action_violations: List of action violations (Drinking/Eating)
        """
        if not self._initialized:
            self.initialize()

        self._frame_count += 1

        # 1. Detect persons with YOLOv8 native tracking
        persons = self.person_detector.detect_with_tracking(frame)

        # 2. Generate/propagate person masks with SAM3 or SAM2
        if self.sam3_segmenter and self._use_sam3:
            persons = self._add_masks_to_persons_sam3(frame, persons)
        elif self.sam2_segmenter and self._use_sam2:
            persons = self._add_masks_to_persons_sam2(frame, persons)

        # 3. Detect PPE and violations (YOLOv11 returns both)
        if self.ppe_detector.model is None:
            logger.error("=" * 80)
            logger.error("YOLOv11 model is None - cannot detect PPE or violations!")
            logger.error("Check model loading logs above for errors.")
            logger.error("=" * 80)

        # Call YOLOv11 detector
        logger.debug(f"Calling YOLOv11 detect() on frame shape: {frame.shape}")
        ppe_result = self.ppe_detector.detect(frame)
        ppe_detections = ppe_result.get("ppe_detections", {})
        violation_detections = ppe_result.get("violation_detections", {})
        action_violations = ppe_result.get("action_violations", [])

        # 4. Generate PPE masks with SAM3 or SAM2 (single-frame mode)
        segmenter = self.sam3_segmenter or self.sam2_segmenter
        if segmenter and self._segment_ppe:
            ppe_detections = self._add_masks_to_ppe(frame, ppe_detections)
            violation_detections = self._add_masks_to_ppe(frame, violation_detections)

        # Log detection results for debugging
        total_violations = sum(len(dets) for dets in violation_detections.values())
        total_ppe = sum(len(dets) for dets in ppe_detections.values())

        # Always log (even if 0) so we can see what's happening
        logger.info(
            f"HybridDetector: {len(persons)} persons, {total_ppe} PPE, "
            f"{total_violations} violations, {len(action_violations)} actions"
        )

        if (
            total_violations == 0
            and total_ppe == 0
            and self.ppe_detector.model is not None
        ):
            logger.warning(
                "YOLOv11 model is loaded but returned 0 detections. "
                "Check confidence thresholds and model output."
            )

        return {
            "persons": persons,
            "ppe_detections": ppe_detections,
            "violation_detections": violation_detections,
            "action_violations": action_violations,
            "frame_shape": frame.shape[:2],
        }

    def _add_masks_to_persons_sam3(
        self, frame: np.ndarray, persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add SAM3 masks to person detections using streaming video.

        SAM3's process_frame() handles everything:
        - Automatic session initialization
        - Adding new objects from detections
        - Mask propagation
        """
        if not persons:
            return persons

        try:
            # SAM3 handles everything in one call
            masks = self.sam3_segmenter.process_frame(frame, persons)

            # Assign masks to persons by track_id
            mask_count = 0
            for person in persons:
                track_id = person.get("track_id")
                if track_id is not None and track_id in masks:
                    mask = masks[track_id]
                    # Ensure mask is uint8 and 2D
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                    person["mask"] = mask
                    mask_count += 1
                    logger.debug(f"SAM3 mask for track {track_id}, shape: {mask.shape}")

            if mask_count > 0:
                logger.info(f"SAM3 assigned {mask_count} masks to persons")

        except Exception as e:
            logger.warning(f"SAM3 mask generation failed: {e}")
            # Try single-frame fallback
            return self._segment_persons_single_frame_sam3(frame, persons)

        return persons

    def _segment_persons_single_frame_sam3(
        self, frame: np.ndarray, persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: segment persons using single-frame SAM3 (no streaming)."""
        if not self.sam3_segmenter:
            return persons

        boxes = [p.get("box") for p in persons if p.get("box")]
        labels = [f"person_{p.get('track_id', i)}" for i, p in enumerate(persons)]

        if not boxes:
            return persons

        try:
            results = self.sam3_segmenter.segment_boxes(frame, boxes, labels)
            for person, result in zip(persons, results):
                if result.get("valid") and result.get("mask") is not None:
                    mask = result["mask"]
                    # Ensure mask is uint8 and 2D
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                    person["mask"] = mask
                    logger.debug(
                        f"SAM3 single-frame: assigned mask to track {person.get('track_id')}"
                    )
        except Exception as e:
            logger.warning(f"SAM3 single-frame segmentation failed: {e}")

        return persons

    def _add_masks_to_persons_sam2(
        self, frame: np.ndarray, persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add SAM2 masks to person detections using video propagation.

        On first frame or new video: initialize video tracking
        On subsequent frames: propagate masks (every N frames)
        For new tracks: add them to SAM2 tracking
        """
        if not persons:
            return persons

        current_track_ids = {
            p.get("track_id") for p in persons if p.get("track_id") is not None
        }

        # Check if this is first frame or new video
        if not self._video_initialized:
            # Initialize video tracking with all current persons
            try:
                self.sam2_segmenter.init_video_tracking(frame, persons)
                self._video_initialized = True
                self._last_track_ids = current_track_ids
                logger.info(
                    f"SAM2 video tracking initialized with {len(persons)} persons"
                )
                # Get initial masks from add_new_object calls in init_video_tracking
                # We need to propagate immediately to get masks
                try:
                    masks = self.sam2_segmenter.propagate_masks(frame)
                    mask_count = 0
                    for person in persons:
                        track_id = person.get("track_id")
                        if track_id in masks:
                            mask = masks[track_id]
                            # Ensure mask is uint8 and 2D
                            if mask.dtype != np.uint8:
                                mask = (mask > 0).astype(np.uint8)
                            if mask.ndim == 3:
                                mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                            person["mask"] = mask
                            mask_count += 1
                            logger.info(
                                f"Assigned initial mask to track {track_id}, shape: {mask.shape}"
                            )
                    if mask_count == 0:
                        logger.warning(
                            "No masks assigned after initialization, falling back to single-frame"
                        )
                        return self._segment_persons_single_frame(frame, persons)
                    else:
                        logger.info(f"Successfully assigned {mask_count} initial masks")
                except Exception as e:
                    logger.warning(f"Failed to get initial masks after init: {e}")
                    # Fall back to single-frame segmentation for initial masks
                    return self._segment_persons_single_frame(frame, persons)
            except Exception as e:
                logger.warning(f"SAM2 video tracking init failed: {e}")
                # Fall back to per-frame segmentation
                return self._segment_persons_single_frame(frame, persons)

        # Find new tracks (not yet in SAM2)
        new_track_ids = current_track_ids - self._last_track_ids
        for person in persons:
            track_id = person.get("track_id")
            if track_id in new_track_ids:
                box = person.get("box")
                if box:
                    try:
                        result = self.sam2_segmenter.add_new_object(
                            frame, box, track_id
                        )
                        if result and result.get("mask") is not None:
                            mask = result["mask"]
                            # Ensure mask is uint8 and 2D
                            if mask.dtype != np.uint8:
                                mask = (mask > 0).astype(np.uint8)
                            if mask.ndim == 3:
                                mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                            person["mask"] = mask
                            logger.info(
                                f"Added new track {track_id} to SAM2 with mask shape: {mask.shape}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to add track {track_id} to SAM2: {e}")

        # Find lost tracks (remove from SAM2)
        lost_track_ids = self._last_track_ids - current_track_ids
        for track_id in lost_track_ids:
            try:
                self.sam2_segmenter.remove_object(track_id)
                logger.debug(f"Removed lost track {track_id} from SAM2")
            except Exception as e:
                logger.warning(f"Failed to remove track {track_id} from SAM2: {e}")

        self._last_track_ids = current_track_ids

        # Propagate masks (every N frames for performance)
        # Always propagate on first frame after initialization to get initial masks
        should_propagate = (
            self._frame_count % self._sam2_propagate_interval == 0
            or not any(p.get("mask") is not None for p in persons)
        )

        if should_propagate:
            try:
                masks = self.sam2_segmenter.propagate_masks(frame)
                # Assign masks to persons by track_id
                mask_count = 0
                for person in persons:
                    track_id = person.get("track_id")
                    if track_id in masks:
                        mask = masks[track_id]
                        # Ensure mask is uint8 and 2D
                        if mask.dtype != np.uint8:
                            mask = (mask > 0).astype(np.uint8)
                        if mask.ndim == 3:
                            mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                        person["mask"] = mask
                        mask_count += 1
                        logger.debug(
                            f"Propagated mask for track {track_id}, shape: {mask.shape}"
                        )
                if mask_count > 0:
                    logger.info(
                        f"Assigned {mask_count} masks to persons via propagation"
                    )
            except Exception as e:
                logger.warning(f"SAM2 mask propagation failed: {e}")
                # Fall back to single-frame segmentation
                return self._segment_persons_single_frame(frame, persons)
        else:
            # On non-propagation frames, ensure persons still have masks from previous propagation
            # If a person lost their mask (e.g., track was lost and recreated), try to get it
            missing_masks = [
                p
                for p in persons
                if p.get("mask") is None and p.get("track_id") is not None
            ]
            if missing_masks:
                logger.debug(
                    f"Found {len(missing_masks)} persons without masks on non-propagation frame"
                )
                # Try to get masks from SAM2's current state
                try:
                    masks = self.sam2_segmenter.propagate_masks(frame)
                    for person in missing_masks:
                        track_id = person.get("track_id")
                        if track_id in masks:
                            mask = masks[track_id]
                            # Ensure mask is uint8 and 2D
                            if mask.dtype != np.uint8:
                                mask = (mask > 0).astype(np.uint8)
                            if mask.ndim == 3:
                                mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                            person["mask"] = mask
                            logger.debug(
                                f"Recovered mask for track {track_id} on non-propagation frame"
                            )
                except Exception as e:
                    logger.debug(
                        f"Could not recover masks on non-propagation frame: {e}"
                    )

        return persons

    def _segment_persons_single_frame(
        self, frame: np.ndarray, persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: segment persons using single-frame SAM2 (no video propagation)."""
        if not self.sam2_segmenter:
            return persons

        boxes = [p.get("box") for p in persons if p.get("box")]
        labels = [f"person_{p.get('track_id', i)}" for i, p in enumerate(persons)]

        if not boxes:
            return persons

        try:
            results = self.sam2_segmenter.segment_boxes(frame, boxes, labels)
            for person, result in zip(persons, results):
                if result.get("valid") and result.get("mask") is not None:
                    mask = result["mask"]
                    # Ensure mask is uint8 and 2D
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                    person["mask"] = mask
                    logger.debug(
                        f"Single-frame segmentation: assigned mask to track {person.get('track_id')}, shape: {mask.shape}"
                    )
        except Exception as e:
            logger.warning(f"SAM2 single-frame segmentation failed: {e}")

        return persons

    def _add_masks_to_ppe(
        self, frame: np.ndarray, ppe_detections: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Add SAM3/SAM2 masks to PPE detections (single-frame, not video propagation)."""
        segmenter = self.sam3_segmenter or self.sam2_segmenter
        if not segmenter:
            return ppe_detections

        # Collect all PPE boxes
        all_boxes = []
        all_labels = []
        box_to_key = []  # (ppe_type, index) for mapping back

        for ppe_type, detections in ppe_detections.items():
            for i, det in enumerate(detections):
                box = det.get("box")
                if box:
                    all_boxes.append(box)
                    all_labels.append(ppe_type)
                    box_to_key.append((ppe_type, i))

        if not all_boxes:
            return ppe_detections

        try:
            results = segmenter.segment_boxes(frame, all_boxes, all_labels)
            for (ppe_type, idx), result in zip(box_to_key, results):
                if result.get("valid") and result.get("mask") is not None:
                    ppe_detections[ppe_type][idx]["mask"] = result["mask"]
        except Exception as e:
            logger.warning(f"PPE segmentation failed: {e}")

        return ppe_detections

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

            # Get person mask if available (for mask-based containment)
            person_mask = person.get("mask")

            # Check for positive PPE detections (e.g., "Googles", "Mask", "Lab Coat")
            for ppe_class, ppe_list in ppe_detections.items():
                # Skip violation classes (they start with "No")
                if ppe_class.startswith("No "):
                    continue

                for ppe in ppe_list:
                    ppe_box = ppe.get("box", [0, 0, 0, 0])
                    ppe_mask = ppe.get("mask")

                    # Use mask-based containment if both masks available, else box-based
                    if person_mask is not None and ppe_mask is not None:
                        containment = calculate_mask_containment(ppe_mask, person_mask)
                    else:
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
                    viol_mask = viol.get("mask")

                    # For small violation boxes, use multiple association methods:
                    # 1. Check if violation box center is inside person box
                    viol_center_x = (viol_box[0] + viol_box[2]) / 2
                    viol_center_y = (viol_box[1] + viol_box[3]) / 2
                    center_inside = (
                        person_box[0] <= viol_center_x <= person_box[2]
                        and person_box[1] <= viol_center_y <= person_box[3]
                    )

                    # 2. Calculate containment - prefer mask-based if available
                    if person_mask is not None and viol_mask is not None:
                        containment = calculate_mask_containment(viol_mask, person_mask)
                        reverse_containment = 0.0  # Not needed for mask-based
                        max_containment = containment
                    else:
                        # Box-based containment (violation box inside person box)
                        containment = calculate_box_containment(viol_box, person_box)
                        # Also check reverse containment (person box inside violation box)
                        reverse_containment = calculate_box_containment(
                            person_box, viol_box
                        )
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
                                f"ppe_type={required_name}"
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
        # Reset internal video tracking state
        self._frame_count = 0
        self._video_initialized = False
        self._last_track_ids = set()

        # Reset SAM3 segmenter if available (preferred)
        if self.sam3_segmenter is not None:
            try:
                if hasattr(self.sam3_segmenter, "reset_video_state"):
                    self.sam3_segmenter.reset_video_state()
                logger.info("SAM3 video state reset")
            except Exception as e:
                logger.warning(f"Failed to reset SAM3 state: {e}")

        # Reset SAM2 segmenter if available (fallback)
        if self.sam2_segmenter is not None:
            try:
                if hasattr(self.sam2_segmenter, "reset_video_state"):
                    self.sam2_segmenter.reset_video_state()
                logger.info("SAM2 video state reset")
            except Exception as e:
                logger.warning(f"Failed to reset SAM2 state: {e}")

        # YOLOv8 tracking is handled internally per video, resets automatically

    def reset_sam_state(self) -> None:
        """Alias for reset_video_state (for explicit SAM reset)."""
        self.reset_video_state()

    def reset_sam2_state(self) -> None:
        """Legacy alias for reset_video_state."""
        self.reset_video_state()

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
