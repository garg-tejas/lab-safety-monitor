"""
Person Detector

Lightweight person detector using pretrained YOLOv8-nano.
Auto-downloads the model on first run (~6MB).
Only detects COCO class 0 (person).
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ..core.config import settings


class PersonDetector:
    """
    Lightweight person detector using pretrained YOLOv8-nano.

    Uses COCO-pretrained model which detects 80 classes,
    but we only extract class 0 (person).
    """

    PERSON_CLASS_ID = 0  # COCO class ID for person

    def __init__(self):
        self.model: Optional[Any] = None
        self._initialized = False
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.device = "cuda" if self._cuda_available() else "cpu"

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self) -> None:
        """
        Initialize the YOLOv8-nano model.
        Auto-downloads if not present.
        """
        if self._initialized:
            return

        if YOLO is None:
            raise ImportError(
                "ultralytics is required for PersonDetector. "
                "Install with: pip install ultralytics"
            )

        # YOLOv8-nano auto-downloads to ~/.cache/ultralytics
        # or we can specify a local path
        model_path = settings.WEIGHTS_DIR / "person_detector" / "yolov8n.pt"

        if model_path.exists():
            print(f"Loading person detector from: {model_path}")
            self.model = YOLO(str(model_path))
        else:
            # Auto-download from ultralytics hub
            print("Downloading YOLOv8-nano model for person detection...")
            self.model = YOLO("yolov8n.pt")

            # Save to local weights directory for future use
            model_path.parent.mkdir(parents=True, exist_ok=True)
            # Note: ultralytics auto-caches, so this is optional

        # Move to appropriate device
        self.model.to(self.device)

        self._initialized = True
        print(f"PersonDetector initialized on {self.device}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in a frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            List of person detections, each with:
                - id: Detection index
                - box: [x1, y1, x2, y2]
                - score: Confidence score
                - label: "person"
                - mask: None (SAM2 will add masks later)
        """
        if not self._initialized:
            self.initialize()

        if self.model is None:
            return []

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
        )

        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Only include person class (should already be filtered, but double-check)
                if cls != self.PERSON_CLASS_ID:
                    continue

                persons.append(
                    {
                        "id": i,
                        "box": xyxy.tolist(),
                        "score": conf,
                        "label": "person",
                        "mask": None,  # Will be filled by SAM2
                    }
                )

        return persons

    def __repr__(self) -> str:
        return f"PersonDetector(initialized={self._initialized}, device={self.device})"


# Singleton instance
_person_detector: Optional[PersonDetector] = None


def get_person_detector() -> PersonDetector:
    """Get singleton PersonDetector instance."""
    global _person_detector
    if _person_detector is None:
        _person_detector = PersonDetector()
    return _person_detector
