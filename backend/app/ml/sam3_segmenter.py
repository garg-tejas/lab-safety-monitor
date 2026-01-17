"""
SAM 3 Segmenter

Wrapper for SAM 3 (Segment Anything Model 3) using ModelScope for model download.
Supports both image segmentation and streaming video tracking.

SAM 3 improvements over SAM 2:
- Native streaming video inference
- Better multi-object tracking
- Improved mask quality

Model download via ModelScope (no HuggingFace auth required):
    modelscope download --model facebook/sam3 sam3/sam3.pt
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from ..core.config import settings

logger = logging.getLogger(__name__)


def download_sam3_from_modelscope(target_path: Path) -> bool:
    """
    Download SAM3 model from ModelScope.

    Args:
        target_path: Path to save the sam3.pt file

    Returns:
        True if download successful, False otherwise
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        logger.error("ModelScope not installed. Install with: pip install modelscope")
        return False

    model_id = getattr(settings, "SAM3_MODEL", "facebook/sam3")

    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading SAM3 from ModelScope: {model_id}")
        print(f"Downloading SAM3 from ModelScope: {model_id}...")

        # Download to cache, then copy specific file
        cache_dir = snapshot_download(
            model_id,
            allow_file_pattern=["sam3/sam3.pt", "sam3.pt"],
        )

        # Find the downloaded file
        cache_path = Path(cache_dir)
        pt_file = None

        for candidate in [
            cache_path / "sam3" / "sam3.pt",
            cache_path / "sam3.pt",
        ]:
            if candidate.exists():
                pt_file = candidate
                break

        if pt_file is None:
            # List files in cache for debugging
            all_files = list(cache_path.rglob("*.pt"))
            logger.warning(f"SAM3 .pt files in cache: {all_files}")
            if all_files:
                pt_file = all_files[0]

        if pt_file and pt_file.exists():
            # Copy or symlink to target path
            import shutil

            shutil.copy2(pt_file, target_path)
            logger.info(f"SAM3 model saved to: {target_path}")
            print(f"SAM3 model saved to: {target_path}")
            return True
        else:
            logger.error("Could not find sam3.pt in downloaded files")
            return False

    except Exception as e:
        logger.error(f"Failed to download SAM3 from ModelScope: {e}")
        print(f"Failed to download SAM3: {e}")
        return False


class SAM3Segmenter:
    """
    SAM 3 wrapper with streaming video support.

    Uses the sam3.pt checkpoint from ModelScope for video tracking
    with box/point prompts.

    Features:
    - Streaming video inference (frame-by-frame)
    - Box-prompted segmentation
    - Multi-object tracking with consistent IDs
    - Automatic mask propagation
    """

    def __init__(self):
        self.model = None
        self.predictor = None
        self._initialized = False
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.dtype = None  # Set during initialization

        # Video state tracking
        self._video_initialized = False
        self._tracked_object_ids: Dict[int, int] = {}  # track_id -> sam3_obj_id
        self._next_obj_id = 1
        self._frame_count = 0
        self._inference_state: Optional[Any] = None

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if torch is None:
            return False
        return torch.cuda.is_available()

    def _get_model_path(self) -> Path:
        """Get SAM3 model path, downloading if necessary."""
        model_path = getattr(settings, "SAM3_MODEL_PATH", None)
        if model_path is None:
            model_path = settings.WEIGHTS_DIR / "sam3" / "sam3.pt"

        if not model_path.exists():
            logger.info(f"SAM3 model not found at {model_path}, downloading...")
            if not download_sam3_from_modelscope(model_path):
                raise FileNotFoundError(
                    f"SAM3 model not found at {model_path} and download failed.\n"
                    f"Manual download:\n"
                    f"  pip install modelscope\n"
                    f"  modelscope download --model facebook/sam3 sam3/sam3.pt --local_dir {model_path.parent}"
                )

        return model_path

    def initialize(self) -> None:
        """Initialize SAM 3 model from local checkpoint."""
        if self._initialized:
            return

        if torch is None:
            raise ImportError(
                "PyTorch is required for SAM3. Install with: pip install torch"
            )

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_path = self._get_model_path()

        print(f"Loading SAM3 model from {model_path}...")
        logger.info(f"Loading SAM3 model from {model_path}")

        try:
            # Try to use SAM3's build function if available (similar to SAM2)
            # SAM3 may have a similar structure to SAM2
            try:
                from sam3.build_sam import build_sam3, build_sam3_video_predictor
                from sam3.sam3_image_predictor import SAM3ImagePredictor

                # Find config file
                config_path = self._find_sam3_config()

                self.model = build_sam3(
                    config_file=config_path,
                    ckpt_path=str(model_path),
                    device=self.device,
                )
                self.predictor = SAM3ImagePredictor(self.model)

                # Try to build video predictor
                try:
                    self.video_predictor = build_sam3_video_predictor(
                        config_file=config_path,
                        ckpt_path=str(model_path),
                        device=self.device,
                    )
                    logger.info("SAM3 video predictor initialized")
                except Exception as e:
                    logger.warning(f"SAM3 video predictor not available: {e}")
                    self.video_predictor = None

            except ImportError:
                # SAM3 package not installed, try loading raw checkpoint
                logger.info("SAM3 package not found, loading raw checkpoint...")
                self._load_raw_checkpoint(model_path)

            self._initialized = True
            print(f"SAM3Segmenter initialized on {self.device} with dtype {self.dtype}")
            logger.info(f"SAM3Segmenter initialized on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise

    def _find_sam3_config(self) -> str:
        """Find SAM3 configuration file."""
        try:
            import sam3

            sam3_dir = Path(sam3.__file__).parent

            config_candidates = [
                sam3_dir / "configs" / "sam3.yaml",
                sam3_dir / "configs" / "sam3_default.yaml",
                sam3_dir / "sam3.yaml",
            ]

            for config in config_candidates:
                if config.exists():
                    return str(config)

            # Return default config name for Hydra
            return "sam3.yaml"

        except ImportError:
            return "sam3.yaml"

    def _load_raw_checkpoint(self, model_path: Path) -> None:
        """Load SAM3 from raw checkpoint without build functions."""
        # Load the checkpoint
        checkpoint = torch.load(
            str(model_path),
            map_location=self.device,
            weights_only=False,
        )

        # Store checkpoint for later use - actual model structure depends on SAM3 format
        self._raw_checkpoint = checkpoint
        self.model = checkpoint.get("model", checkpoint)

        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()

        logger.info("Loaded SAM3 raw checkpoint (limited functionality)")
        print("SAM3 loaded from raw checkpoint - some features may be limited")

    def segment_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment objects using box prompts.

        Args:
            frame: BGR numpy array from OpenCV
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
            if hasattr(self, "predictor") and self.predictor is not None:
                # Use SAM3ImagePredictor
                self.predictor.set_image(frame_rgb)

                for box, label in zip(boxes, labels):
                    box_np = np.array(box, dtype=np.float32)

                    masks, scores, _ = self.predictor.predict(
                        box=box_np,
                        multimask_output=True,
                    )

                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    score = float(scores[best_idx])

                    density = self._calculate_mask_density(mask, box)
                    density_threshold = getattr(settings, "MASK_DENSITY_THRESHOLD", 0.1)

                    results.append(
                        {
                            "mask": mask.astype(np.uint8),
                            "box": box,
                            "label": label,
                            "score": score,
                            "density": density,
                            "valid": density >= density_threshold,
                        }
                    )
            else:
                # Fallback: return box-based masks
                h, w = frame.shape[:2]
                for box, label in zip(boxes, labels):
                    mask = self._box_to_mask(box, h, w)
                    results.append(
                        {
                            "mask": mask,
                            "box": box,
                            "label": label,
                            "score": 0.5,
                            "density": 1.0,
                            "valid": True,
                        }
                    )

        except Exception as e:
            logger.warning(f"SAM3 segmentation failed: {e}")
            # Fallback to box masks
            h, w = frame.shape[:2]
            for box, label in zip(boxes, labels):
                mask = self._box_to_mask(box, h, w)
                results.append(
                    {
                        "mask": mask,
                        "box": box,
                        "label": label,
                        "score": 0.0,
                        "density": 1.0,
                        "valid": False,
                    }
                )

        return results

    def _box_to_mask(self, box: List[float], h: int, w: int) -> np.ndarray:
        """Create a simple box mask as fallback."""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in box]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask[y1:y2, x1:x2] = 1
        return mask

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

    def init_video_session(self, frame: Optional[np.ndarray] = None) -> None:
        """
        Initialize a streaming video session.

        Args:
            frame: Optional first frame to initialize with
        """
        if not self._initialized:
            self.initialize()

        # Reset state
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0

        if hasattr(self, "video_predictor") and self.video_predictor is not None:
            try:
                if frame is not None:
                    frame_rgb = frame[:, :, ::-1].copy()
                    self._inference_state = self.video_predictor.init_state(frame_rgb)
                else:
                    self._inference_state = None

                self._video_initialized = True
                logger.info("SAM3 streaming video session initialized")
            except Exception as e:
                logger.warning(f"Failed to init SAM3 video session: {e}")
                self._video_initialized = False
        else:
            # Video predictor not available, mark as initialized anyway
            # We'll fall back to per-frame segmentation
            self._video_initialized = True
            logger.info("SAM3 video session initialized (per-frame mode)")

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
            self.init_video_session(frame)

        result = {}

        # If we have a proper video predictor, use it
        if (
            hasattr(self, "video_predictor")
            and self.video_predictor is not None
            and self._inference_state is not None
        ):
            try:
                result = self._process_frame_video(frame, detections)
            except Exception as e:
                logger.warning(f"Video processing failed, using per-frame: {e}")
                result = self._process_frame_single(frame, detections)
        else:
            # Fall back to per-frame segmentation
            result = self._process_frame_single(frame, detections)

        self._frame_count += 1
        return result

    def _process_frame_video(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]],
    ) -> Dict[int, np.ndarray]:
        """Process frame using video predictor."""
        frame_rgb = frame[:, :, ::-1].copy()

        # Add new objects if provided
        if detections:
            for det in detections:
                track_id = det.get("track_id")
                box = det.get("box")

                if track_id is not None and box is not None:
                    if track_id not in self._tracked_object_ids:
                        self._add_object_video(box, track_id)

        # Propagate to this frame
        out_obj_ids, out_mask_logits = self.video_predictor.propagate_in_video(
            inference_state=self._inference_state,
            frame_idx=self._frame_count,
            image=frame_rgb,
        )

        # Map back to track IDs
        result = {}
        obj_id_to_track = {v: k for k, v in self._tracked_object_ids.items()}

        for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
            track_id = obj_id_to_track.get(int(obj_id))
            if track_id is not None:
                mask = (mask_logits > 0).cpu().numpy().astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask[0]
                result[track_id] = mask

        return result

    def _add_object_video(self, box: List[float], track_id: int) -> None:
        """Add object to video tracking."""
        obj_id = self._next_obj_id
        self._next_obj_id += 1
        self._tracked_object_ids[track_id] = obj_id

        box_np = np.array(box, dtype=np.float32)

        try:
            self.video_predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=self._frame_count,
                obj_id=obj_id,
                box=box_np,
            )
            logger.debug(f"Added track {track_id} as SAM3 object {obj_id}")
        except Exception as e:
            logger.warning(f"Failed to add object {track_id}: {e}")
            del self._tracked_object_ids[track_id]

    def _process_frame_single(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]],
    ) -> Dict[int, np.ndarray]:
        """Process frame using per-frame segmentation."""
        if not detections:
            return {}

        result = {}

        # Extract boxes and track_ids
        boxes = []
        track_ids = []

        for det in detections:
            track_id = det.get("track_id")
            box = det.get("box")
            if track_id is not None and box is not None:
                boxes.append(box)
                track_ids.append(track_id)

        if boxes:
            seg_results = self.segment_boxes(frame, boxes)
            for track_id, seg in zip(track_ids, seg_results):
                if seg.get("mask") is not None:
                    result[track_id] = seg["mask"]

        return result

    def remove_object(self, track_id: int) -> None:
        """Remove an object from tracking."""
        if track_id in self._tracked_object_ids:
            del self._tracked_object_ids[track_id]
            logger.debug(f"Removed track {track_id} from SAM3 tracking")

    def reset_video_state(self) -> None:
        """Reset video tracking state for a new video."""
        self._video_initialized = False
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0
        self._inference_state = None

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
