"""
YOLOv11 Detector for PPE Detection

Uses trained YOLOv11 model for PPE detection with violation classes.
Classes (12): Gloves, Googles, Head Mask, Lab Coat, Mask,
              No Gloves, No googles, No Head Mask, No Lab coat, No Mask,
              Drinking, Eating
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from ..core.config import settings

logger = logging.getLogger(__name__)


class YOLOv11Detector:
    """PPE Detector using trained YOLOv11 model."""

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

    CLASS_MAPPING = {
        2: "gloves",
        3: "safety goggles",
        4: "head mask",
        5: "lab coat",
        6: "face mask",
    }

    VIOLATION_CLASS_MAPPING = {
        7: "gloves",
        8: "head mask",
        9: "lab coat",
        10: "face mask",
        11: "safety goggles",
    }

    ACTION_VIOLATIONS = {0: "drinking", 1: "eating"}

    def __init__(self):
        self.model = None
        self.model_type = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.violation_threshold = getattr(
            settings, "VIOLATION_CONFIDENCE_THRESHOLD", 0.3
        )
        self._initialized = False

        self.multi_scale_enabled = getattr(settings, "MULTI_SCALE_ENABLED", True)
        self.multi_scale_factors = getattr(
            settings, "MULTI_SCALE_FACTORS", [1.0, 1.5, 2.0]
        )
        self.multi_scale_nms_threshold = getattr(
            settings, "MULTI_SCALE_NMS_THRESHOLD", 0.5
        )

    def _check_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self):
        """Initialize YOLOv11 model."""
        if self._initialized:
            return

        model_path = settings.YOLOV11_MODEL_PATH
        if not model_path:
            logger.warning("YOLOV11_MODEL_PATH not set - using mock detector")
            self._initialized = True
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"YOLOv11 model not found at {model_path}")
            self._initialized = True
            return

        try:
            if model_path.suffix == ".onnx":
                self._load_onnx_model(model_path)
            else:
                self._load_pytorch_model(model_path)

            self._initialized = True
            logger.info(f"YOLOv11 loaded: {model_path.name} ({self.model_type})")
            if self.multi_scale_enabled:
                logger.info(
                    f"Multi-scale detection enabled: {self.multi_scale_factors}"
                )

        except Exception as e:
            logger.error(f"YOLOv11 loading failed: {e}")
            self._initialized = True

    def _load_pytorch_model(self, model_path: Path):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model_type = "pytorch"

        if hasattr(self.model, "names") and self.model.names:
            for class_id, class_name in self.model.names.items():
                if class_id in self.CLASS_NAMES:
                    self.CLASS_NAMES[class_id] = class_name

    def _load_onnx_model(self, model_path: Path):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.model = ort.InferenceSession(str(model_path), providers=providers)
        self.model_type = "onnx"

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect PPE items and violations in a frame."""
        if not self._initialized:
            self.initialize()

        results = {
            "persons": [],
            "ppe_detections": {},
            "violation_detections": {},
            "action_violations": [],
            "frame_shape": frame.shape[:2],
        }

        if self.model is None:
            return self._mock_detect(frame)

        try:
            if self.model_type == "pytorch":
                detections = self._detect_pytorch(frame)
            else:
                detections = self._detect_onnx(frame)

            persons, ppe_detections, violation_detections, action_violations = (
                self._parse_detections(detections, frame.shape[:2])
            )

            results["persons"] = persons
            results["ppe_detections"] = ppe_detections
            results["violation_detections"] = violation_detections
            results["action_violations"] = action_violations

        except Exception as e:
            logger.error(f"YOLOv11 detection error: {e}")
            return self._mock_detect(frame)

        return results

    def _detect_pytorch(self, frame: np.ndarray) -> List[Dict]:
        if self.multi_scale_enabled and len(self.multi_scale_factors) > 1:
            return self._detect_multiscale(frame)
        return self._detect_single_scale(frame, scale=1.0)

    def _detect_single_scale(self, frame: np.ndarray, scale: float = 1.0) -> List[Dict]:
        h, w = frame.shape[:2]

        if scale != 1.0:
            scaled_frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_frame = frame

        min_threshold = min(self.confidence_threshold, self.violation_threshold)
        results = self.model(scaled_frame, conf=min_threshold, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
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
        all_detections = []
        for scale in self.multi_scale_factors:
            all_detections.extend(self._detect_single_scale(frame, scale))

        if not all_detections:
            return []

        return self._apply_nms(all_detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

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

            boxes = np.array([d["box"] for d in class_dets])
            scores = np.array([d["confidence"] for d in class_dets])
            keep_indices = self._nms(boxes, scores, self.multi_scale_nms_threshold)
            for idx in keep_indices:
                merged.append(class_dets[idx])

        return merged

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> List[int]:
        if len(boxes) == 0:
            return []

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            remaining = order[1:]
            ious = self._compute_iou_batch(boxes[i], boxes[remaining])
            order = remaining[ious < threshold]

        return keep

    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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
        input_name = self.model.get_inputs()[0].name
        input_shape = self.model.get_inputs()[0].shape
        img_size = input_shape[2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_array = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        outputs = self.model.run(None, {input_name: img_array})
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0]

        detections = []
        h, w = frame.shape[:2]
        scale_x, scale_y = w / img_size, h / img_size
        min_threshold = min(self.confidence_threshold, self.violation_threshold)

        for detection in output:
            if len(detection) < 6:
                continue

            x1, y1, x2, y2, conf = [float(d) for d in detection[:5]]

            if abs(x1) < 1.0 and abs(x2) < 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

            if len(detection) == 6:
                class_id = int(detection[5])
                final_conf = conf
            else:
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                final_conf = float(conf * class_scores[class_id])

            if (
                0 <= class_id <= 11
                and 0 <= final_conf <= 1
                and final_conf >= min_threshold
            ):
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": final_conf,
                        "box": [x1, y1, x2, y2],
                    }
                )

        return detections

    def _parse_detections(
        self, detections: List[Dict], frame_shape: Tuple[int, int]
    ) -> Tuple:
        persons = []
        ppe_detections = {ppe_type: [] for ppe_type in settings.PPE_PROMPTS}
        violation_detections = {}
        action_violations = []

        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            box = det["box"]

            if class_id in self.CLASS_MAPPING:
                ppe_type = self.CLASS_MAPPING[class_id]
                if confidence >= self.confidence_threshold:
                    if ppe_type not in ppe_detections:
                        ppe_detections[ppe_type] = []
                    ppe_detections[ppe_type].append(
                        {"box": box, "score": confidence, "mask": None}
                    )

            elif class_id in self.VIOLATION_CLASS_MAPPING:
                if confidence >= self.violation_threshold:
                    ppe_type = self.VIOLATION_CLASS_MAPPING[class_id]
                    if ppe_type not in violation_detections:
                        violation_detections[ppe_type] = []
                    violation_detections[ppe_type].append(
                        {
                            "box": box,
                            "score": confidence,
                            "mask": None,
                            "class_name": self.CLASS_NAMES.get(class_id, ""),
                        }
                    )

            elif class_id in self.ACTION_VIOLATIONS:
                action_violations.append(
                    {
                        "box": box,
                        "score": confidence,
                        "action": self.ACTION_VIOLATIONS[class_id],
                        "class_name": self.CLASS_NAMES.get(class_id, ""),
                    }
                )

        # Create person boxes from detections
        all_boxes = []
        for ppe_list in ppe_detections.values():
            all_boxes.extend([d["box"] for d in ppe_list])
        for viol_list in violation_detections.values():
            all_boxes.extend([d["box"] for d in viol_list])
        for action in action_violations:
            all_boxes.append(action["box"])

        if all_boxes:
            persons = self._create_person_boxes(all_boxes, frame_shape)

        return persons, ppe_detections, violation_detections, action_violations

    def _create_person_boxes(self, boxes: List, frame_shape: tuple) -> List[Dict]:
        if not boxes:
            return []

        x1_min = min(box[0] for box in boxes)
        y1_min = min(box[1] for box in boxes)
        x2_max = max(box[2] for box in boxes)
        y2_max = max(box[3] for box in boxes)

        h, w = frame_shape
        padding = 50
        person_box = [
            max(0, x1_min - padding),
            max(0, y1_min - padding),
            min(w, x2_max + padding),
            min(h, y2_max + padding),
        ]

        return [{"id": 0, "box": person_box, "score": 0.9, "mask": None}]

    def _mock_detect(self, frame: np.ndarray) -> Dict[str, Any]:
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
        ppe_detections: Dict,
        violation_detections: Optional[Dict] = None,
        action_violations: Optional[List] = None,
    ) -> List[Dict]:
        """Associate PPE and violations with persons."""
        required_ppe = set(settings.REQUIRED_PPE)
        violation_detections = violation_detections or {}
        action_violations = action_violations or []

        for person in persons:
            person["detected_ppe"] = []
            person["missing_ppe"] = []
            person["action_violations"] = []
            person["detection_confidence"] = {}
            person_box = person["box"]

            for ppe_type, detections in ppe_detections.items():
                for detection in detections:
                    if self._boxes_overlap(person_box, detection["box"]):
                        if ppe_type not in person["detected_ppe"]:
                            person["detected_ppe"].append(ppe_type)
                            person["detection_confidence"][ppe_type] = detection.get(
                                "score", 0.0
                            )
                        break

            for ppe_type, detections in violation_detections.items():
                for detection in detections:
                    if self._boxes_overlap(person_box, detection["box"]):
                        if (
                            ppe_type not in person["missing_ppe"]
                            and ppe_type in required_ppe
                        ):
                            person["missing_ppe"].append(ppe_type)
                            person["detection_confidence"][f"no_{ppe_type}"] = (
                                detection.get("score", 0.0)
                            )
                        break

            for action in action_violations:
                if self._boxes_overlap(person_box, action["box"]):
                    person["action_violations"].append(
                        {"action": action["action"], "score": action["score"]}
                    )

            person["is_violation"] = (
                len(person["missing_ppe"]) > 0 or len(person["action_violations"]) > 0
            )

        return persons

    def _boxes_overlap(
        self, box1: List[float], box2: List[float], threshold: float = 0.3
    ) -> bool:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return box2_area > 0 and intersection / box2_area >= threshold


# Singleton
_yolov11_detector = None


def get_yolov11_detector() -> YOLOv11Detector:
    global _yolov11_detector
    if _yolov11_detector is None:
        _yolov11_detector = YOLOv11Detector()
    return _yolov11_detector
