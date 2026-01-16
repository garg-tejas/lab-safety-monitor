"""
YOLOv11 Detector for PPE Detection

Uses trained YOLOv11 model for PPE detection with violation classes.
Supports both PyTorch (.pt) and ONNX (.onnx) model formats.

Classes (12):
- PPE Present: Gloves, Googles, Head Mask, Lab Coat, Mask
- PPE Missing: No Gloves, No googles, No Head Mask, No Lab coat, No Mask
- Actions: Drinking, Eating
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from ..core.config import settings

logger = logging.getLogger(__name__)


class YOLOv11Detector:
    """
    PPE Detector using trained YOLOv11 model.

    The model directly detects both PPE presence and absence:
    - Positive classes: Gloves, Googles, Head Mask, Lab Coat, Mask
    - Violation classes: No Gloves, No googles, No Head Mask, No Lab coat, No Mask
    - Action classes: Drinking, Eating (violations in lab)
    """

    # YOLOv11 model classes (from training) - ACTUAL ORDER from model.names
    # Index: Class Name
    # 0: Drinking, 1: Eating, 2: Gloves, 3: Googles, 4: Head Mask,
    # 5: Lab Coat, 6: Mask, 7: No Gloves, 8: No Head Mask, 9: No Lab coat,
    # 10: No Mask, 11: No googles
    CLASS_NAMES = {
        0: "Drinking",
        1: "Eating",
        2: "Gloves",
        3: "Googles",
        4: "Head Mask",
        5: "Lab Coat",
        6: "Mask",
        7: "No Gloves",
        8: "No Head Mask",
        9: "No Lab coat",
        10: "No Mask",
        11: "No googles",
    }

    # Class mapping from model class ID to system PPE types (positive detections only)
    CLASS_MAPPING = {
        2: "gloves",  # Gloves
        3: "safety goggles",  # Googles -> safety goggles
        4: "head mask",  # Head Mask -> head mask
        5: "lab coat",  # Lab Coat -> lab coat
        6: "face mask",  # Mask -> face mask
    }

    # Violation class mapping - these directly indicate missing PPE
    # IMPORTANT: Order matches actual model output (model.names)
    VIOLATION_CLASS_MAPPING = {
        7: "gloves",  # No Gloves -> missing gloves
        8: "head mask",  # No Head Mask -> missing head mask
        9: "lab coat",  # No Lab coat -> missing lab coat
        10: "face mask",  # No Mask -> missing face mask
        11: "safety goggles",  # No googles -> missing safety goggles
    }

    # Action violations (not PPE but safety violations)
    ACTION_VIOLATIONS = {
        0: "drinking",  # Drinking in lab
        1: "eating",  # Eating in lab
    }

    # Reverse mapping for person detection (if person class exists)
    PERSON_CLASS_ID = None  # Will be set if person detection is available

    def __init__(self):
        self.model = None
        self.model_type = None  # 'pytorch' or 'onnx'
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.violation_threshold = getattr(
            settings, "VIOLATION_CONFIDENCE_THRESHOLD", 0.3
        )
        self._initialized = False

        # Multi-scale detection settings
        self.multi_scale_enabled = getattr(settings, "MULTI_SCALE_ENABLED", True)
        self.multi_scale_factors = getattr(
            settings, "MULTI_SCALE_FACTORS", [1.0, 1.5, 2.0]
        )
        self.multi_scale_nms_threshold = getattr(
            settings, "MULTI_SCALE_NMS_THRESHOLD", 0.5
        )

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self):
        """Lazy initialization of YOLOv11 model."""
        if self._initialized:
            return

        model_path = settings.YOLOV11_MODEL_PATH

        if not model_path:
            logger.error("YOLOV11_MODEL_PATH is not set in configuration!")
            logger.warning("Falling back to mock detector for development")
            self._initialized = True
            return

        model_path = Path(model_path)
        logger.info("=" * 80)
        logger.info(f"🔍 YOLOv11 Model Loading:")
        logger.info(f"  Requested path: {model_path.absolute()}")
        logger.info(f"  File exists: {model_path.exists()}")
        logger.info(f"  File extension: {model_path.suffix}")
        if model_path.suffix == ".onnx":
            logger.warning("⚠️ ONNX model detected - parsing may be problematic!")
            logger.warning(
                "⚠️ Consider using PyTorch (.pt) model for better compatibility"
            )
        logger.info("=" * 80)

        if not model_path.exists():
            logger.error(f"YOLOv11 model not found at {model_path.absolute()}")
            logger.error(f"Model path exists check: {model_path.exists()}")
            logger.error(f"Parent directory exists: {model_path.parent.exists()}")
            logger.error(
                f"Parent directory contents: {list(model_path.parent.glob('*')) if model_path.parent.exists() else 'N/A'}"
            )
            logger.warning("Falling back to mock detector for development")
            self._initialized = True
            return

        try:
            if model_path.suffix == ".onnx":
                self._load_onnx_model(model_path)
            else:
                self._load_pytorch_model(model_path)

            self._initialized = True
            logger.info("=" * 80)
            logger.info(f"✓ YOLOv11 model loaded successfully!")
            logger.info(f"  Type: {self.model_type}")
            logger.info(f"  Path: {model_path.absolute()}")
            if self.multi_scale_enabled:
                logger.info(
                    f"  Multi-scale: ENABLED (scales: {self.multi_scale_factors})"
                )
            else:
                logger.info(f"  Multi-scale: DISABLED")
            logger.info("=" * 80)

            # Verify model works with a test frame
            try:
                import numpy as np

                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_result = self.detect(test_frame)
                logger.info(
                    f"✓ YOLOv11 model verification test passed - model is ready"
                )
            except Exception as e:
                logger.warning(f"⚠️ YOLOv11 model verification test failed: {e}")
        except Exception as e:
            logger.error(f"YOLOv11 model loading error: {e}", exc_info=True)
            logger.warning("Falling back to mock detector for development")
            self._initialized = True  # Use mock mode

    def _load_pytorch_model(self, model_path: Path):
        """Load PyTorch YOLOv11 model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLOv11 PyTorch model from {model_path}...")
            self.model = YOLO(str(model_path))
            self.model_type = "pytorch"

            # Use model's class names if available (more accurate than hardcoded)
            if hasattr(self.model, "names") and self.model.names:
                logger.info(f"Model class names: {list(self.model.names.values())}")
                # Update CLASS_NAMES with model's actual names
                for class_id, class_name in self.model.names.items():
                    if class_id in self.CLASS_NAMES:
                        self.CLASS_NAMES[class_id] = class_name
                        logger.debug(f"Updated class {class_id}: {class_name}")

            logger.info(f"YOLOv11 PyTorch model loaded on {self.device}")

            # Verify model works with a test detection (optional, can be disabled)
            # This helps catch model loading issues early
            try:
                import numpy as np

                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(test_frame, verbose=False)
                logger.debug("YOLOv11 model verification test passed")
            except Exception as e:
                logger.warning(f"YOLOv11 model verification test failed: {e}")
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}", exc_info=True)
            raise

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX YOLOv11 model."""
        try:
            import onnxruntime as ort

            logger.info(f"Loading YOLOv11 ONNX model from {model_path}...")

            # Set providers based on device
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]

            self.model = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )
            self.model_type = "onnx"
            logger.info(f"YOLOv11 ONNX model loaded with providers: {providers}")

            # Log model input/output details
            input_name = self.model.get_inputs()[0].name
            input_shape = self.model.get_inputs()[0].shape
            logger.debug(f"ONNX model input: {input_name}, shape: {input_shape}")
        except ImportError:
            raise ImportError(
                "onnxruntime package not installed. Install with: pip install onnxruntime-gpu"
            )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect PPE items and violations in a frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            Dict with detected PPE items, violations, their boxes, and scores
        """
        if not self._initialized:
            self.initialize()

        results = {
            "persons": [],
            "ppe_detections": {},
            "violation_detections": {},  # Direct "No X" class detections
            "action_violations": [],  # Drinking/Eating violations
            "frame_shape": frame.shape[:2],
        }

        if self.model is None:
            # Mock mode for development
            logger.warning(
                "YOLOv11 model is None - using mock detector. Check model loading."
            )
            return self._mock_detect(frame)

        try:
            if self.model_type == "pytorch":
                detections = self._detect_pytorch(frame)
            else:
                detections = self._detect_onnx(frame)

            # Log raw detections with class names for debugging
            if len(detections) > 0:
                logger.info(f"🔍 YOLOv11 raw detections ({len(detections)} total):")
                for i, det in enumerate(detections):
                    class_id = det["class_id"]
                    class_name = self.CLASS_NAMES.get(
                        class_id, f"unknown_class_{class_id}"
                    )
                    confidence = det["confidence"]
                    box = det["box"]
                    logger.info(
                        f"  Detection {i + 1}: class_id={class_id}, class_name='{class_name}', "
                        f"confidence={confidence:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
                    )
            else:
                logger.warning("⚠️ YOLOv11 returned 0 raw detections")

            # Separate persons, PPE, violations, and actions
            persons, ppe_detections, violation_detections, action_violations = (
                self._parse_detections(detections, frame.shape[:2])
            )

            # Debug logging - use INFO level so it's visible
            total_ppe = sum(len(dets) for dets in ppe_detections.values())
            total_violations = sum(len(dets) for dets in violation_detections.values())

            # Always log YOLOv11 detection results (even if 0)
            logger.info(
                f"🔍 YOLOv11 detect() called: {len(detections)} raw detections, "
                f"{total_ppe} PPE items, {total_violations} violations, "
                f"{len(action_violations)} action violations"
            )

            if total_violations > 0:
                logger.info(
                    f"✓ Violations detected: {dict((k, len(v)) for k, v in violation_detections.items() if v)}"
                )
            if action_violations:
                logger.info(
                    f"✓ Action violations: {[a.get('action') for a in action_violations]}"
                )

            if len(detections) == 0:
                logger.warning(
                    "⚠️ YOLOv11 returned 0 detections - model may not be working correctly"
                )
            elif total_ppe == 0 and total_violations == 0:
                logger.warning(
                    f"⚠️ YOLOv11 found {len(detections)} detections but parsed to 0 PPE/violations. "
                    f"Check class mapping and confidence thresholds."
                )

            results["persons"] = persons
            results["ppe_detections"] = ppe_detections
            results["violation_detections"] = violation_detections
            results["action_violations"] = action_violations

        except Exception as e:
            logger.error(f"YOLOv11 detection error: {e}", exc_info=True)
            return self._mock_detect(frame)

        return results

    def _detect_pytorch(self, frame: np.ndarray) -> List[Dict]:
        """Run detection using PyTorch model."""
        # Use multi-scale detection if enabled
        if self.multi_scale_enabled and len(self.multi_scale_factors) > 1:
            return self._detect_multiscale(frame)

        # Single-scale detection
        return self._detect_single_scale(frame, scale=1.0)

    def _detect_single_scale(self, frame: np.ndarray, scale: float = 1.0) -> List[Dict]:
        """Run detection at a single scale."""
        h, w = frame.shape[:2]

        # Resize frame if scale != 1.0
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_frame = frame

        # Use lower threshold for initial detection to catch violations
        min_threshold = min(self.confidence_threshold, self.violation_threshold)
        results = self.model(scaled_frame, conf=min_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Convert box to native Python floats (not np.float64)
                xyxy = box.xyxy[0].cpu().numpy()

                # Scale coordinates back to original frame size
                if scale != 1.0:
                    xyxy = xyxy / scale

                detections.append(
                    {
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "box": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                    }
                )

        return detections

    def _detect_multiscale(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection at multiple scales and merge results.

        This improves detection of small objects like goggles by running
        the detector on upscaled versions of the image.
        """
        all_detections = []

        for scale in self.multi_scale_factors:
            logger.debug(f"Multi-scale detection at scale {scale}x")
            scale_detections = self._detect_single_scale(frame, scale)
            all_detections.extend(scale_detections)
            logger.debug(f"  Found {len(scale_detections)} detections at {scale}x")

        if not all_detections:
            return []

        logger.info(
            f"Multi-scale detection: {len(all_detections)} total detections "
            f"from {len(self.multi_scale_factors)} scales"
        )

        # Apply NMS to merge overlapping detections from different scales
        merged_detections = self._apply_nms(all_detections)

        logger.info(
            f"After NMS: {len(merged_detections)} detections "
            f"(merged from {len(all_detections)})"
        )

        return merged_detections

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to merge overlapping detections.

        Groups detections by class and applies NMS within each class.
        """
        if not detections:
            return []

        # Group detections by class
        class_detections: Dict[int, List[Dict]] = {}
        for det in detections:
            class_id = det["class_id"]
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)

        merged = []

        for class_id, class_dets in class_detections.items():
            if len(class_dets) == 1:
                merged.append(class_dets[0])
                continue

            # Extract boxes and scores for NMS
            boxes = np.array([d["box"] for d in class_dets])
            scores = np.array([d["confidence"] for d in class_dets])

            # Apply NMS
            keep_indices = self._nms(boxes, scores, self.multi_scale_nms_threshold)

            for idx in keep_indices:
                merged.append(class_dets[idx])

        return merged

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> List[int]:
        """
        Non-Maximum Suppression implementation.

        Args:
            boxes: Array of boxes [[x1, y1, x2, y2], ...]
            scores: Array of confidence scores
            threshold: IoU threshold for suppression

        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []

        # Sort by score (descending)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Take the highest scoring box
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = self._compute_iou_batch(boxes[i], boxes[remaining])

            # Keep boxes with IoU below threshold
            mask = ious < threshold
            order = remaining[mask]

        return keep

    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and an array of boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-6)

    def _detect_onnx(self, frame: np.ndarray) -> List[Dict]:
        """Run detection using ONNX model."""
        # Get input details
        input_name = self.model.get_inputs()[0].name
        input_shape = self.model.get_inputs()[0].shape
        img_size = input_shape[2]  # Assuming square input (e.g., 640)

        # Preprocess image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_array = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        outputs = self.model.run(None, {input_name: img_array})

        # Log output shapes for debugging
        logger.info(f"🔍 ONNX model outputs: {len(outputs)} outputs")
        for i, out in enumerate(outputs):
            logger.info(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
            if out.size > 0 and out.size < 100:  # Log small outputs
                logger.info(f"    Sample values: {out.flatten()[:10]}")

        # YOLOv11 ONNX typically outputs: [batch, num_detections, 6]
        # Format: [x1, y1, x2, y2, confidence, class_id]
        # OR: [batch, num_detections, 4+num_classes] with raw scores
        # OR: [batch, num_anchors, 4+1+num_classes] (raw predictions before NMS)

        output = outputs[
            0
        ]  # Shape: [batch, num_detections, features] or [batch, num_anchors, features]

        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]  # [num_detections, features] or [num_anchors, features]

        logger.info(
            f"🔍 Processing ONNX output: shape={output.shape}, first detection sample={output[0] if len(output) > 0 else 'empty'}"
        )

        detections = []
        h, w = frame.shape[:2]
        scale_x = w / img_size
        scale_y = h / img_size

        # Use lower threshold to catch violations, filter by class in _parse_detections
        min_threshold = min(self.confidence_threshold, self.violation_threshold)

        for detection in output:
            if len(detection) < 6:
                # Skip if not enough elements
                continue

            # Check output format
            # Format 1: [x1, y1, x2, y2, conf, class_id] - 6 elements
            # Format 2: [x1, y1, x2, y2, conf, class0_score, class1_score, ...] - 4+num_classes elements

            x1, y1, x2, y2 = (
                float(detection[0]),
                float(detection[1]),
                float(detection[2]),
                float(detection[3]),
            )
            conf = float(detection[4])

            # Check if coordinates are normalized (0-1) or absolute
            if abs(x1) < 1.0 and abs(y1) < 1.0 and abs(x2) < 1.0 and abs(y2) < 1.0:
                # Normalized coordinates, convert to absolute
                x1 = x1 * w
                y1 = y1 * h
                x2 = x2 * w
                y2 = y2 * h
            else:
                # Absolute coordinates, scale to original image
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y

            # Determine format and extract class_id and confidence
            if len(detection) == 6:
                # Format 1: [x1, y1, x2, y2, conf, class_id]
                class_id = int(detection[5])
                final_conf = conf
            else:
                # Format 2: [x1, y1, x2, y2, conf, class0_score, class1_score, ...]
                # Get class scores (remaining elements after [x1, y1, x2, y2, conf])
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                class_conf = float(class_scores[class_id])
                # Use combined confidence (objectness * class probability)
                final_conf = float(conf * class_conf)

            # Validate class_id is in expected range (0-11 for 12 classes)
            if class_id < 0 or class_id > 11:
                logger.warning(
                    f"⚠️ Invalid class_id={class_id} from ONNX model. "
                    f"Expected 0-11. Detection: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, "
                    f"conf={conf:.3f}, final_conf={final_conf:.3f}"
                )
                continue

            # Validate confidence is reasonable (0-1)
            if final_conf < 0 or final_conf > 1:
                logger.warning(
                    f"⚠️ Invalid confidence={final_conf:.3f} from ONNX model. "
                    f"Expected 0-1. Detection: class_id={class_id}, x1={x1:.1f}, y1={y1:.1f}, "
                    f"x2={x2:.1f}, y2={y2:.1f}, raw_conf={conf:.3f}"
                )
                continue

            # Filter by threshold
            if final_conf >= min_threshold:
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": final_conf,
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        return detections

    def _parse_detections(
        self, detections: List[Dict], frame_shape: Tuple[int, int]
    ) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, List[Dict]], List[Dict]]:
        """
        Parse detections into persons, PPE items, violations, and action violations.

        Args:
            detections: List of detection dicts with class_id, confidence, box
            frame_shape: (height, width) of frame

        Returns:
            Tuple of (persons_list, ppe_detections_dict, violation_detections_dict, action_violations_list)
        """
        persons = []
        ppe_detections = {ppe_type: [] for ppe_type in settings.PPE_PROMPTS}
        violation_detections = {}  # Direct violation detections from "No X" classes
        action_violations = []  # Drinking/Eating violations

        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            box = det["box"]
            class_name = self.CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")

            # Check if this is a positive PPE detection
            if class_id in self.CLASS_MAPPING:
                ppe_type = self.CLASS_MAPPING[class_id]
                if confidence >= self.confidence_threshold:
                    if ppe_type not in ppe_detections:
                        ppe_detections[ppe_type] = []
                    ppe_detections[ppe_type].append(
                        {
                            "box": box,
                            "score": confidence,
                            "mask": None,  # YOLOv11 doesn't provide masks
                        }
                    )
                    logger.info(
                        f"✓ PPE detected: class_id={class_id}, class_name='{class_name}', "
                        f"ppe_type={ppe_type}, confidence={confidence:.3f}"
                    )
                else:
                    logger.warning(
                        f"✗ PPE filtered (low confidence): class_id={class_id}, "
                        f"class_name='{class_name}', confidence={confidence:.3f} < "
                        f"threshold={self.confidence_threshold}"
                    )

            # Check if this is a violation detection (No X classes)
            elif class_id in self.VIOLATION_CLASS_MAPPING:
                # Apply violation-specific threshold (lower than general threshold)
                if confidence >= self.violation_threshold:
                    ppe_type = self.VIOLATION_CLASS_MAPPING[class_id]
                    if ppe_type not in violation_detections:
                        violation_detections[ppe_type] = []
                    violation_detections[ppe_type].append(
                        {
                            "box": box,
                            "score": confidence,
                            "mask": None,
                            "class_name": class_name,
                        }
                    )
                    logger.info(
                        f"✓ Violation detected: class_id={class_id}, class_name='{class_name}', "
                        f"ppe_type={ppe_type}, confidence={confidence:.3f}, "
                        f"box={[round(b, 1) for b in box]}"
                    )
                else:
                    logger.warning(
                        f"✗ Violation filtered (low confidence): class_id={class_id}, "
                        f"class_name='{class_name}', confidence={confidence:.3f} < "
                        f"threshold={self.violation_threshold}"
                    )

            # Check if this is an action violation (Drinking/Eating)
            elif class_id in self.ACTION_VIOLATIONS:
                action_type = self.ACTION_VIOLATIONS[class_id]
                action_violations.append(
                    {
                        "box": box,
                        "score": confidence,
                        "action": action_type,
                        "class_name": class_name,
                    }
                )
                logger.info(
                    f"✓ Action violation detected: class_id={class_id}, class_name='{class_name}', "
                    f"action={action_type}, confidence={confidence:.3f}, "
                    f"box={[round(b, 1) for b in box]}"
                )
            else:
                # Unknown class - log it
                logger.warning(
                    f"⚠️ Unknown class detected: class_id={class_id}, class_name='{class_name}', "
                    f"confidence={confidence:.3f}. Not in CLASS_MAPPING, VIOLATION_CLASS_MAPPING, "
                    f"or ACTION_VIOLATIONS."
                )

        # For person detection, we'll use a simple heuristic:
        # If we detect PPE items or violations, assume there's a person nearby
        all_boxes = []
        for ppe_list in ppe_detections.values():
            all_boxes.extend([d["box"] for d in ppe_list])
        for viol_list in violation_detections.values():
            all_boxes.extend([d["box"] for d in viol_list])
        for action in action_violations:
            all_boxes.append(action["box"])

        if all_boxes:
            persons = self._create_person_boxes_from_ppe(
                {"all": [{"box": b} for b in all_boxes]}, frame_shape
            )

        return persons, ppe_detections, violation_detections, action_violations

    def _create_person_boxes_from_ppe(
        self, ppe_detections: Dict[str, List[Dict]], frame_shape: tuple
    ) -> List[Dict]:
        """
        Create person bounding boxes from PPE detections.

        This is a simple heuristic - in production, you might want to use
        a separate person detector or YOLOv11 with person class.
        """
        persons = []
        all_ppe_boxes = []

        # Collect all PPE boxes
        for ppe_type, detections in ppe_detections.items():
            for det in detections:
                all_ppe_boxes.append(det["box"])

        if not all_ppe_boxes:
            return persons

        # Group boxes by spatial proximity
        # Simple approach: find bounding box that contains all PPE items
        if all_ppe_boxes:
            x1_min = min(box[0] for box in all_ppe_boxes)
            y1_min = min(box[1] for box in all_ppe_boxes)
            x2_max = max(box[2] for box in all_ppe_boxes)
            y2_max = max(box[3] for box in all_ppe_boxes)

            # Expand box to include person (add padding)
            padding = 50
            h, w = frame_shape
            person_box = [
                max(0, x1_min - padding),
                max(0, y1_min - padding),
                min(w, x2_max + padding),
                min(h, y2_max + padding),
            ]

            persons.append(
                {
                    "id": 0,
                    "box": person_box,
                    "score": 0.9,  # High confidence since we detected PPE
                    "mask": None,
                }
            )

        return persons

    def _mock_detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Mock detection for development without YOLOv11."""
        h, w = frame.shape[:2]

        return {
            "persons": [
                {"id": 0, "box": [100, 50, 300, 400], "score": 0.95, "mask": None}
            ],
            "ppe_detections": {
                "safety goggles": [],
                "face mask": [
                    {"box": [150, 200, 250, 280], "score": 0.8, "mask": None}
                ],
                "lab coat": [
                    {"box": [100, 100, 300, 400], "score": 0.85, "mask": None}
                ],
                "gloves": [],
                "head mask": [],
            },
            "violation_detections": {
                "safety goggles": [
                    {
                        "box": [150, 100, 250, 180],
                        "score": 0.75,
                        "mask": None,
                        "class_name": "no safety goggles",
                    }
                ],
            },
            "action_violations": [],
            "frame_shape": (h, w),
        }

    def associate_ppe_to_persons(
        self,
        persons: List[Dict],
        ppe_detections: Dict[str, List[Dict]],
        violation_detections: Optional[Dict[str, List[Dict]]] = None,
        action_violations: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Associate detected PPE items and violations with persons based on spatial overlap.

        With YOLOv11's new violation classes, we can directly detect missing PPE
        instead of inferring it from absence of positive detections.

        Args:
            persons: List of person detections
            ppe_detections: Dict of PPE type -> list of positive detections
            violation_detections: Dict of PPE type -> list of "No X" detections
            action_violations: List of action violations (Drinking/Eating)

        Returns:
            List of persons with their associated PPE and violations
        """
        required_ppe = set(settings.REQUIRED_PPE)
        violation_detections = violation_detections or {}
        action_violations = action_violations or []

        for person in persons:
            person["detected_ppe"] = []
            person["missing_ppe"] = []
            person["action_violations"] = []
            person["detection_confidence"] = {}
            person_box = person["box"]

            # Check for positive PPE detections
            for ppe_type, detections in ppe_detections.items():
                for detection in detections:
                    ppe_box = detection["box"]
                    if self._boxes_overlap(person_box, ppe_box):
                        if ppe_type not in person["detected_ppe"]:
                            person["detected_ppe"].append(ppe_type)
                            person["detection_confidence"][ppe_type] = float(
                                detection.get("score", 0.0)
                            )
                        break

            # Check for direct violation detections (No X classes)
            for ppe_type, detections in violation_detections.items():
                for detection in detections:
                    viol_box = detection["box"]
                    if self._boxes_overlap(person_box, viol_box):
                        # Direct detection of missing PPE
                        if (
                            ppe_type not in person["missing_ppe"]
                            and ppe_type in required_ppe
                        ):
                            person["missing_ppe"].append(ppe_type)
                            person["detection_confidence"][f"no_{ppe_type}"] = float(
                                detection.get("score", 0.0)
                            )
                        break

            # Check for action violations (Drinking/Eating)
            for action in action_violations:
                action_box = action["box"]
                if self._boxes_overlap(person_box, action_box):
                    person["action_violations"].append(
                        {
                            "action": action["action"],
                            "score": action["score"],
                        }
                    )

            # For PPE types in required_ppe that we didn't detect (neither positive nor negative),
            # we can't assume they're missing without explicit detection
            # The YOLOv11 model directly detects "No X" so we rely on that

            person["is_violation"] = (
                len(person["missing_ppe"]) > 0 or len(person["action_violations"]) > 0
            )

        return persons

    def _boxes_overlap(
        self, box1: List[float], box2: List[float], threshold: float = 0.3
    ) -> bool:
        """Check if two boxes overlap with IoU above threshold."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Check if PPE is mostly inside person box
        if box2_area > 0:
            overlap_ratio = intersection / box2_area
            return overlap_ratio >= threshold

        return False


# Singleton instance
_yolov11_detector = None


def get_yolov11_detector() -> YOLOv11Detector:
    """Get singleton YOLOv11 detector instance."""
    global _yolov11_detector
    if _yolov11_detector is None:
        _yolov11_detector = YOLOv11Detector()
    return _yolov11_detector
