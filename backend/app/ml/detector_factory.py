"""
Detector Factory

Factory pattern to create and return the appropriate detector based on configuration.
"""

from typing import Union
from ..core.config import settings
from .sam3_detector import SAM3Detector, get_detector as get_sam3_detector
from .yolov11_detector import YOLOv11Detector, get_yolov11_detector

_mock_detector = None


def get_detector() -> Union[SAM3Detector, YOLOv11Detector, "HybridDetector"]:
    """
    Get the appropriate detector based on configuration.

    Returns:
        Detector instance (HybridDetector, YOLOv11Detector, or SAM3Detector)
    """
    detector_type = settings.DETECTOR_TYPE.lower()

    if detector_type == "mock" or settings.USE_MOCK_DETECTOR:
        global _mock_detector
        if _mock_detector is None:
            _mock_detector = SAM3Detector(mock_mode=True)
        return _mock_detector

    if detector_type == "hybrid":
        from .hybrid_detector import get_hybrid_detector

        return get_hybrid_detector()
    elif detector_type == "yolov11":
        return get_yolov11_detector()
    elif detector_type == "sam3":
        return get_sam3_detector()
    else:
        raise ValueError(f"Unknown DETECTOR_TYPE: {detector_type}")


# Type alias for the detector union
DetectorType = Union[SAM3Detector, YOLOv11Detector, "HybridDetector"]
