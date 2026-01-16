"""
SAM 3 Segmenter

Wrapper for SAM 3 (Segment Anything Model 3) using HuggingFace Transformers.
Supports both image segmentation and streaming video tracking.

SAM 3 improvements over SAM 2:
- Native streaming video inference
- Text-prompted segmentation
- Better multi-object tracking
- Simpler API via Transformers
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from ..core.config import settings

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """
    SAM 3 wrapper with streaming video support.

    Uses Sam3TrackerVideo for video tracking (box/point prompts)
    and Sam3Video for text-prompted segmentation.

    Features:
    - Streaming video inference (frame-by-frame)
    - Box-prompted segmentation
    - Multi-object tracking with consistent IDs
    - Automatic mask propagation
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.inference_session = None
        self._initialized = False
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Video state tracking
        self._video_initialized = False
        self._tracked_object_ids: Dict[int, int] = {}  # track_id -> sam3_obj_id
        self._next_obj_id = 1
        self._frame_count = 0

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if torch is None:
            return False
        return torch.cuda.is_available()

    def initialize(self) -> None:
        """Initialize SAM 3 model from HuggingFace."""
        if self._initialized:
            return

        if torch is None:
            raise ImportError(
                "PyTorch is required for SAM3. Install with: pip install torch"
            )

        try:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        except ImportError:
            raise ImportError(
                "SAM3 requires transformers>=4.47.0. "
                "Install with: pip install transformers>=4.47.0"
            )

        model_name = getattr(settings, "SAM3_MODEL", "facebook/sam3")

        print(f"Loading SAM3 model from {model_name}...")
        logger.info(f"Loading SAM3 model from {model_name}")

        try:
            self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_name)
            self.model = Sam3TrackerVideoModel.from_pretrained(model_name).to(
                self.device, dtype=self.dtype
            )
            self.model.eval()

            self._initialized = True
            print(f"SAM3Segmenter initialized on {self.device} with dtype {self.dtype}")
            logger.info(f"SAM3Segmenter initialized on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise

    def init_video_session(self) -> None:
        """
        Initialize a streaming video session.

        Call this once at the start of video processing.
        """
        if not self._initialized:
            self.initialize()

        # Initialize streaming session (no pre-loaded frames)
        self.inference_session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )

        self._video_initialized = True
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0

        logger.info("SAM3 streaming video session initialized")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a single frame with optional new detections.

        Args:
            frame: BGR numpy array from OpenCV
            detections: Optional list of new detections to track
                Each detection should have 'track_id' and 'box' keys

        Returns:
            Dict mapping track_id -> binary mask
        """
        if not self._initialized:
            self.initialize()

        if not self._video_initialized:
            self.init_video_session()

        # Convert BGR to RGB
        frame_rgb = frame[:, :, ::-1].copy()

        # Process frame through processor
        inputs = self.processor(
            images=frame_rgb, device=self.device, return_tensors="pt"
        )

        # Add new objects if provided
        if detections:
            for det in detections:
                track_id = det.get("track_id")
                box = det.get("box")

                if track_id is not None and box is not None:
                    # Only add if not already tracking
                    if track_id not in self._tracked_object_ids:
                        self._add_object(track_id, box, inputs.original_sizes[0])

        # Run inference on this frame
        with torch.no_grad():
            outputs = self.model(
                inference_session=self.inference_session,
                frame=inputs.pixel_values[0],
            )

        self._frame_count += 1

        # Post-process masks
        if outputs.pred_masks is not None:
            masks = self.processor.post_process_masks(
                [outputs.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )[0]

            # Map SAM3 object IDs back to track IDs
            result = {}
            sam3_to_track = {v: k for k, v in self._tracked_object_ids.items()}

            for i, obj_id in enumerate(self.inference_session.obj_ids):
                track_id = sam3_to_track.get(obj_id)
                if track_id is not None and i < len(masks):
                    mask = masks[i].cpu().numpy()
                    # Ensure mask is 2D
                    if mask.ndim == 3:
                        mask = mask[0]
                    result[track_id] = mask.astype(np.uint8)

            return result

        return {}

    def _add_object(
        self,
        track_id: int,
        box: List[float],
        original_size: Tuple[int, int],
    ) -> None:
        """Add a new object to track using box prompt."""
        obj_id = self._next_obj_id
        self._next_obj_id += 1
        self._tracked_object_ids[track_id] = obj_id

        # Format box as expected by SAM3: [[[[x1, y1, x2, y2]]]]
        input_boxes = [[[[box[0], box[1], box[2], box[3]]]]]

        try:
            self.processor.add_inputs_to_inference_session(
                inference_session=self.inference_session,
                frame_idx=self._frame_count,
                obj_ids=obj_id,
                input_boxes=input_boxes,
                original_size=original_size,
            )
            logger.debug(f"Added track {track_id} as SAM3 object {obj_id}")
        except Exception as e:
            logger.warning(f"Failed to add object {track_id}: {e}")
            del self._tracked_object_ids[track_id]

    def remove_object(self, track_id: int) -> None:
        """Remove an object from tracking."""
        if track_id in self._tracked_object_ids:
            # SAM3 doesn't have explicit remove, just stop tracking
            del self._tracked_object_ids[track_id]
            logger.debug(f"Removed track {track_id} from SAM3 tracking")

    def segment_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects using box prompts (single-frame, no tracking).

        Args:
            frame: BGR numpy array
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            labels: Optional list of labels

        Returns:
            List of segmentation results with masks
        """
        if not self._initialized:
            self.initialize()

        if not boxes:
            return []

        if labels is None:
            labels = [f"object_{i}" for i in range(len(boxes))]

        # Convert BGR to RGB
        frame_rgb = frame[:, :, ::-1].copy()

        results = []

        try:
            # Use Sam3Tracker for single-image segmentation
            from transformers import Sam3TrackerProcessor, Sam3TrackerModel

            # Use cached single-image model if available
            if not hasattr(self, "_image_model"):
                model_name = getattr(settings, "SAM3_MODEL", "facebook/sam3")
                self._image_processor = Sam3TrackerProcessor.from_pretrained(model_name)
                self._image_model = Sam3TrackerModel.from_pretrained(model_name).to(
                    self.device
                )
                self._image_model.eval()

            # Process each box
            for i, (box, label) in enumerate(zip(boxes, labels)):
                input_boxes = [[[[box[0], box[1], box[2], box[3]]]]]

                inputs = self._image_processor(
                    images=frame_rgb,
                    input_boxes=input_boxes,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._image_model(**inputs, multimask_output=False)

                masks = self._image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"],
                )[0]

                # Get best mask
                mask = masks[0, 0].numpy().astype(np.uint8)

                # Calculate density for validation
                density = self._calculate_mask_density(mask, box)
                density_threshold = getattr(settings, "MASK_DENSITY_THRESHOLD", 0.1)

                results.append(
                    {
                        "mask": mask,
                        "box": box,
                        "label": label,
                        "score": float(outputs.iou_scores[0, 0, 0].cpu())
                        if hasattr(outputs, "iou_scores")
                        else 0.9,
                        "density": density,
                        "valid": density >= density_threshold,
                    }
                )

        except Exception as e:
            logger.warning(f"SAM3 single-frame segmentation failed: {e}")
            # Return empty results for each box
            for box, label in zip(boxes, labels):
                results.append(
                    {
                        "mask": None,
                        "box": box,
                        "label": label,
                        "score": 0.0,
                        "density": 0.0,
                        "valid": False,
                    }
                )

        return results

    def _calculate_mask_density(self, mask: np.ndarray, box: List[float]) -> float:
        """Calculate mask density within bounding box."""
        x1, y1, x2, y2 = [int(c) for c in box]

        h, w = mask.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            return 0.0

        mask_region = mask[y1:y2, x1:x2]
        mask_pixels = np.sum(mask_region > 0)

        return float(mask_pixels) / float(box_area)

    def reset_video_state(self) -> None:
        """Reset video tracking state for a new video."""
        if self.inference_session is not None:
            try:
                self.inference_session.reset_inference_session()
            except Exception:
                pass

        self._video_initialized = False
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0
        self.inference_session = None

        logger.info("SAM3 video state reset")

    def is_tracking(self, track_id: int) -> bool:
        """Check if a track is being tracked."""
        return track_id in self._tracked_object_ids

    def get_tracked_ids(self) -> List[int]:
        """Get list of currently tracked IDs."""
        return list(self._tracked_object_ids.keys())

    def __repr__(self) -> str:
        return (
            f"SAM3Segmenter(initialized={self._initialized}, "
            f"device={self.device}, "
            f"video_initialized={self._video_initialized}, "
            f"tracked_objects={len(self._tracked_object_ids)})"
        )


# Singleton instance
_sam3_segmenter: Optional[SAM3Segmenter] = None


def get_sam3_segmenter() -> SAM3Segmenter:
    """Get singleton SAM3Segmenter instance."""
    global _sam3_segmenter
    if _sam3_segmenter is None:
        _sam3_segmenter = SAM3Segmenter()
    return _sam3_segmenter
