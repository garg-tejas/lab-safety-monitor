"""
YOLOv8 Detector for PPE Detection

Uses trained YOLOv8 model for PPE detection.
Supports both PyTorch (.pt) and ONNX (.onnx) model formats.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from ..core.config import settings


class YOLOv8Detector:
    """
    PPE Detector using trained YOLOv8 model.

    Detects: Gloves, Googles, Lab coat, Mask (from Safety Lab dataset)
    Maps to system PPE types: safety goggles, face mask, lab coat
    """

    # Class mapping from YOLOv8 model to system PPE types
    # Safety Lab dataset classes: 0=Gloves, 1=Googles, 2=Lab coat, 3=Mask
    CLASS_MAPPING = {
        0: "gloves",  # Not in standard PPE, but detected
        1: "safety goggles",  # Googles -> safety goggles
        2: "lab coat",  # Direct mapping
        3: "face mask",  # Mask -> face mask
    }

    # Reverse mapping for person detection (if person class exists)
    PERSON_CLASS_ID = None  # Will be set if person detection is available

    def __init__(self):
        self.model = None
        self.model_type = None  # 'pytorch' or 'onnx'
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self._initialized = False

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self):
        """Lazy initialization of YOLOv8 model."""
        if self._initialized:
            return

        model_path = settings.YOLOV8_MODEL_PATH

        if not model_path or not Path(model_path).exists():
            print(f"YOLOv8 model not found at {model_path}")
            print("Falling back to mock detector for development")
            self._initialized = True
            return

        try:
            if model_path.suffix == ".onnx":
                self._load_onnx_model(model_path)
            else:
                self._load_pytorch_model(model_path)

            self._initialized = True
            print(f"YOLOv8 model loaded successfully! (Type: {self.model_type})")
        except Exception as e:
            print(f"YOLOv8 model loading error: {e}")
            print("Falling back to mock detector for development")
            self._initialized = True  # Use mock mode

    def _load_pytorch_model(self, model_path: Path):
        """Load PyTorch YOLOv8 model."""
        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv8 PyTorch model from {model_path}...")
            self.model = YOLO(str(model_path))
            self.model_type = "pytorch"
            print(f"YOLOv8 PyTorch model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX YOLOv8 model."""
        try:
            import onnxruntime as ort

            print(f"Loading YOLOv8 ONNX model from {model_path}...")

            # Set providers based on device
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]

            self.model = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )
            self.model_type = "onnx"
            print(f"YOLOv8 ONNX model loaded with providers: {providers}")
        except ImportError:
            raise ImportError(
                "onnxruntime package not installed. Install with: pip install onnxruntime-gpu"
            )

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect PPE items in a frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            Dict with detected PPE items, their boxes, and scores
        """
        if not self._initialized:
            self.initialize()

        results = {
            "persons": [],
            "ppe_detections": {},
            "frame_shape": frame.shape[:2],
        }

        if self.model is None:
            # Mock mode for development
            return self._mock_detect(frame)

        try:
            if self.model_type == "pytorch":
                detections = self._detect_pytorch(frame)
            else:
                detections = self._detect_onnx(frame)

            # Separate persons and PPE
            persons, ppe_detections = self._parse_detections(detections, frame.shape)

            results["persons"] = persons
            results["ppe_detections"] = ppe_detections

        except Exception as e:
            print(f"YOLOv8 detection error: {e}")
            return self._mock_detect(frame)

        return results

    def _detect_pytorch(self, frame: np.ndarray) -> List[Dict]:
        """Run detection using PyTorch model."""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append(
                    {
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "box": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    }
                )

        return detections

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

        # YOLOv8 ONNX output format: [batch, num_detections, 4+num_classes]
        # Format: [x_center, y_center, width, height, conf, class0_score, class1_score, ...]
        # Or: [x1, y1, x2, y2, conf, class0_score, class1_score, ...]
        output = outputs[0][0]  # Remove batch dimension: [num_detections, features]

        detections = []
        h, w = frame.shape[:2]
        scale_x = w / img_size
        scale_y = h / img_size

        for detection in output:
            if len(detection) < 5:
                continue

            # Try format: [x_center, y_center, width, height, conf, class_scores...]
            # Or: [x1, y1, x2, y2, conf, class_scores...]
            x1, y1, x2, y2 = detection[0:4]
            conf = detection[4]

            # Check if coordinates are normalized (0-1) or absolute
            if x1 < 1.0 and y1 < 1.0 and x2 < 1.0 and y2 < 1.0:
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

            # Get class scores (remaining elements after [x1, y1, x2, y2, conf])
            if len(detection) > 5:
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                class_conf = float(class_scores[class_id])
                # Use combined confidence
                final_conf = float(conf * class_conf)
            else:
                # No class scores, use detection confidence
                class_id = 0
                final_conf = float(conf)

            if final_conf >= self.confidence_threshold:
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
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Parse detections into persons and PPE items.

        Args:
            detections: List of detection dicts with class_id, confidence, box
            frame_shape: (height, width) of frame

        Returns:
            Tuple of (persons_list, ppe_detections_dict)
        """
        persons = []
        ppe_detections = {ppe_type: [] for ppe_type in settings.PPE_PROMPTS}

        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            box = det["box"]

            # Map class_id to PPE type
            ppe_type = self.CLASS_MAPPING.get(class_id)

            if ppe_type:
                # Add to PPE detections
                if ppe_type in ppe_detections:
                    ppe_detections[ppe_type].append(
                        {
                            "box": box,
                            "score": confidence,
                            "mask": None,  # YOLOv8 doesn't provide masks
                        }
                    )
                elif ppe_type == "gloves":
                    # Gloves detected but not in standard PPE list
                    # Could add to a custom PPE list or ignore
                    pass

        # For person detection, we'll use a simple heuristic:
        # If we detect PPE items, assume there's a person nearby
        # Or use a separate person detector
        # For now, create person boxes from PPE detections
        if ppe_detections:
            # Group PPE detections by spatial proximity to form person boxes
            persons = self._create_person_boxes_from_ppe(ppe_detections, frame_shape)

        return persons, ppe_detections

    def _create_person_boxes_from_ppe(
        self, ppe_detections: Dict[str, List[Dict]], frame_shape: tuple
    ) -> List[Dict]:
        """
        Create person bounding boxes from PPE detections.

        This is a simple heuristic - in production, you might want to use
        a separate person detector or YOLOv8 with person class.
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
        """Mock detection for development without YOLOv8."""
        h, w = frame.shape[:2]

        return {
            "persons": [
                {"id": 0, "box": [100, 50, 300, 400], "score": 0.95, "mask": None}
            ],
            "ppe_detections": {
                "safety goggles": [],
                "protective helmet": [],
                "face mask": [
                    {"box": [150, 200, 250, 280], "score": 0.8, "mask": None}
                ],
                "lab coat": [
                    {"box": [100, 100, 300, 400], "score": 0.85, "mask": None}
                ],
                "safety shoes": [],
            },
            "frame_shape": (h, w),
        }

    def associate_ppe_to_persons(
        self, persons: List[Dict], ppe_detections: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """
        Associate detected PPE items with persons based on spatial overlap.

        Args:
            persons: List of person detections
            ppe_detections: Dict of PPE type -> list of detections

        Returns:
            List of persons with their associated PPE
        """
        required_ppe = set(settings.REQUIRED_PPE)

        for person in persons:
            person["detected_ppe"] = []
            person["missing_ppe"] = []
            person["detection_confidence"] = {}
            person_box = person["box"]

            for ppe_type, detections in ppe_detections.items():
                ppe_found = False

                for detection in detections:
                    ppe_box = detection["box"]

                    # Check if PPE overlaps with person
                    if self._boxes_overlap(person_box, ppe_box):
                        person["detected_ppe"].append(ppe_type)
                        person["detection_confidence"][ppe_type] = float(
                            detection.get("score", 0.0)
                        )
                        ppe_found = True
                        break

                if not ppe_found and ppe_type in required_ppe:
                    person["missing_ppe"].append(ppe_type)
                    person["detection_confidence"].setdefault(ppe_type, 0.0)

            person["is_violation"] = len(person["missing_ppe"]) > 0

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
_yolov8_detector = None


def get_yolov8_detector() -> YOLOv8Detector:
    """Get singleton YOLOv8 detector instance."""
    global _yolov8_detector
    if _yolov8_detector is None:
        _yolov8_detector = YOLOv8Detector()
    return _yolov8_detector
