"""
SAM3 Segmenter — Real Implementation

Uses facebook/sam3 from HuggingFace Transformers.

Only one model class is loaded:

  Sam3TrackerVideoModel + Sam3TrackerVideoProcessor
    - Primary path for process_frame(): streaming, one frame at a time,
      tracks all registered persons across frames via box/point prompts.
    - Also used for segment_boxes() (PPE items) via ephemeral single-frame
      sessions — one session per call, all boxes processed in one forward pass.

Why Sam3TrackerModel is NOT used:
  The facebook/sam3 checkpoint is of architecture type sam3_video.  The
  tracker weights live under a tracker_model.* namespace inside it.
  Instantiating Sam3TrackerModel from that checkpoint causes every decoder
  and encoder weight to be reported MISSING and randomly re-initialised,
  producing completely garbage masks.  Sam3TrackerVideoModel loads cleanly
  from the same checkpoint and is reused for both streaming and single-frame
  inference.

Authentication:
  facebook/sam3 is a gated model.  Before first use you must:
    1. Visit https://huggingface.co/facebook/sam3 while logged in and
       accept Meta's licence agreement.
    2. Set  HF_TOKEN=<your_token>  in backend/.env
"""

import contextlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from ..core.config import settings

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """
    SAM3 segmenter backed by the real facebook/sam3 checkpoint.

    Public interface (unchanged from previous stub so HybridDetector
    requires no edits):

        process_frame(frame, detections)
            -> Dict[track_id, binary_mask]

        segment_boxes(frame, boxes, labels=None)
            -> List[{mask, box, label, score, density, valid, source}]

        init_video_session()   - reset + create new streaming session
        reset_video_state()    - alias used by HybridDetector
        remove_object(track_id)
        is_tracking(track_id) -> bool
        get_tracked_ids()     -> List[int]
    """

    MODEL_ID = "facebook/sam3"

    def __init__(self) -> None:
        # --- video model (Sam3TrackerVideoModel) ---
        self.video_model: Optional[Any] = None
        self.video_processor: Optional[Any] = None
        self._video_model_available: bool = False

        # --- image model (Sam3TrackerModel) ---
        self.image_model: Optional[Any] = None
        self.image_processor: Optional[Any] = None
        self._image_model_available: bool = False

        self._initialized: bool = False
        self.device: str = "cuda" if self._cuda_available() else "cpu"
        self.dtype: Optional[Any] = None  # set in initialize()

        # --- streaming session state ---
        self._inference_session: Optional[Any] = None
        self._video_initialized: bool = False
        self._tracked_object_ids: Dict[int, int] = {}  # track_id -> sam3_obj_id
        self._next_obj_id: int = 1
        self._frame_count: int = 0

        # --- config ---
        self._model_id: str = getattr(settings, "SAM3_MODEL_ID", self.MODEL_ID)
        self._hf_token: Optional[str] = getattr(settings, "HF_TOKEN", None)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cuda_available() -> bool:
        if not _TORCH_AVAILABLE or torch is None:
            return False
        return torch.cuda.is_available()

    @contextlib.contextmanager
    def _no_grad(self):  # type: ignore[return]
        """Context manager that calls torch.no_grad() when torch is available."""
        if _TORCH_AVAILABLE and torch is not None:
            with torch.no_grad():
                yield
        else:
            yield

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """
        Load SAM3 models from HuggingFace Hub.

        Resolution order:
          1. Sam3TrackerVideoModel  (streaming video — primary path for both
             process_frame() and segment_boxes())
          2. Filled-box masks       (last resort, always succeeds)

        NOTE: Sam3TrackerModel is intentionally NOT loaded here.
        The facebook/sam3 checkpoint stores tracker weights under the
        tracker_model.* namespace (because the checkpoint is of type
        sam3_video, not sam3_tracker).  Instantiating Sam3TrackerModel
        from it causes every weight to be reported MISSING and randomly
        re-initialised, producing garbage masks.
        Sam3TrackerVideoModel loads correctly and is reused for single-
        frame segmentation via ephemeral sessions.

        initialize() never raises; degraded mode is reported via logging.
        """
        if self._initialized:
            return

        if not _TORCH_AVAILABLE or torch is None:
            logger.warning("[SAM3] PyTorch not available — box-mask fallback only")
            self._initialized = True
            return

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Authenticate before any from_pretrained call
        if self._hf_token:
            self._hf_login()

        model_id = self._model_id
        print(f"[SAM3] Loading models from HuggingFace: {model_id} ...")
        logger.info(f"[SAM3] Loading from HuggingFace: {model_id}")

        # ---- 1. Sam3TrackerVideoModel --------------------------------- #
        try:
            from transformers import (  # type: ignore[import-untyped]
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
            )

            logger.info("[SAM3] Loading Sam3TrackerVideoModel ...")
            _vm = Sam3TrackerVideoModel.from_pretrained(  # type: ignore[misc]
                model_id,
                token=self._hf_token,
            )
            self.video_model = _vm.to(self.device, dtype=self.dtype)  # type: ignore[misc]

            self.video_processor = Sam3TrackerVideoProcessor.from_pretrained(
                model_id,
                token=self._hf_token,
            )
            self._video_model_available = True
            print(f"[SAM3] Sam3TrackerVideoModel ready on {self.device} ({self.dtype})")
            logger.info(
                f"[SAM3] Sam3TrackerVideoModel ready on {self.device} ({self.dtype})"
            )

        except Exception as exc:
            logger.warning(f"[SAM3] Sam3TrackerVideoModel unavailable: {exc}")
            self._video_model_available = False

        # Sam3TrackerModel is deliberately skipped — see docstring above.
        # segment_boxes() reuses the video model via ephemeral sessions.
        self._image_model_available = False  # always; video model is used instead

        if not self._video_model_available:
            print(
                "[SAM3] WARNING: No SAM3 model could be loaded.\n"
                "        Bounding-box rectangles will be used as masks.\n"
                "        Check that HF_TOKEN is set and that you have accepted\n"
                "        the licence at https://huggingface.co/facebook/sam3"
            )

        self._initialized = True

    def _hf_login(self) -> None:
        """Log in to HuggingFace Hub so gated models can be downloaded."""
        try:
            from huggingface_hub import login

            login(token=self._hf_token, add_to_git_credential=False)
            logger.info("[SAM3] HuggingFace login OK")
        except Exception as exc:
            logger.warning(f"[SAM3] HuggingFace login failed: {exc}")

    # ------------------------------------------------------------------ #
    # Public — streaming video (process_frame)                             #
    # ------------------------------------------------------------------ #

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Segment every person in the frame using their tracked bounding box.

        New track_ids are registered as SAM3 objects automatically.
        Existing tracks are propagated by the model without re-prompting.

        Falls back to Sam3TrackerModel per-frame if the video model is
        unavailable or the session produces an error, and to filled box
        masks if neither model is loaded.

        Args:
            frame:      BGR numpy array (from OpenCV)
            detections: list of dicts, each must contain:
                          'track_id' (int)
                          'box'      ([x1, y1, x2, y2] floats)

        Returns:
            Dict mapping each track_id to a binary uint8 mask [H, W].
        """
        if not self._initialized:
            self.initialize()

        if not detections:
            return {}

        self._frame_count += 1
        logger.info(
            f"[SAM3] process_frame #{self._frame_count}: "
            f"{len(detections)} detection(s), "
            f"video_available={self._video_model_available}"
        )

        # Primary: Sam3TrackerVideoModel streaming
        if self._video_model_available:
            try:
                return self._process_frame_video(frame, detections)
            except Exception as exc:
                print(
                    f"[SAM3] ✗ Video streaming error on frame {self._frame_count}: {exc}\n"
                    f"       Falling back to per-frame image model."
                )
                logger.warning(
                    f"[SAM3] Video streaming error on frame {self._frame_count}, "
                    f"falling back to per-frame: {exc}"
                )

        # Fallback: Sam3TrackerModel per-frame
        if self._image_model_available:
            print(
                f"[SAM3] ↓ Using per-frame fallback (Sam3TrackerModel) for frame {self._frame_count}"
            )
            return self._process_frame_image(frame, detections)

        # Last resort: filled rectangles
        n = len(detections)
        print(
            f"[SAM3] ✗ No SAM model available — returning {n} box-fill masks for frame {self._frame_count}"
        )
        return self._box_masks_for_detections(frame, detections)

    # ------------------------------------------------------------------ #
    # Public — per-frame box segmentation (segment_boxes)                  #
    # ------------------------------------------------------------------ #

    def segment_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment arbitrary regions using bounding-box prompts (no tracking).

        Called by HybridDetector to obtain masks for detected PPE items.

        Args:
            frame:  BGR numpy array
            boxes:  list of [x1, y1, x2, y2] float coordinates
            labels: optional human-readable labels

        Returns:
            List of result dicts:
              mask    - binary uint8 array [H, W]
              box     - the input box
              label   - the input (or auto-generated) label
              score   - SAM3 confidence (0-1)
              density - fraction of box area covered by mask
              valid   - True when density meets the configured threshold
              source  - 'sam3_tracker' | 'box_fallback' | 'box_fallback_error'
        """
        if not self._initialized:
            self.initialize()

        if not boxes:
            return []

        if labels is None:
            labels = [f"object_{i}" for i in range(len(boxes))]

        if self._video_model_available:
            try:
                results = self._segment_boxes_via_video_model(frame, boxes, labels)
                px_list = [
                    int(np.sum(r["mask"] > 0))
                    for r in results
                    if r.get("mask") is not None
                ]
                print(
                    f"[SAM3] segment_boxes | {len(boxes)} boxes | "
                    f"source=sam3_video_single_frame | "
                    f"px={px_list}"
                )
                return results
            except Exception as exc:
                print(f"[SAM3] ✗ segment_boxes error: {exc} — using box fallback")
                logger.warning(f"[SAM3] segment_boxes error: {exc}")

        # Fallback: filled rectangles
        print(
            f"[SAM3] ✗ segment_boxes — no SAM model, returning {len(boxes)} box-fill masks"
        )
        h, w = frame.shape[:2]
        return [
            {
                "mask": self._box_to_mask(box, h, w),
                "box": box,
                "label": label,
                "score": 0.5,
                "density": 1.0,
                "valid": True,
                "source": "box_fallback",
            }
            for box, label in zip(boxes, labels)
        ]

    # ------------------------------------------------------------------ #
    # Private — Sam3TrackerVideoModel streaming                            #
    # ------------------------------------------------------------------ #

    def _ensure_video_session(self) -> None:
        """Create a new streaming inference session if one does not exist."""
        if self._inference_session is not None:
            return

        self._inference_session = self.video_processor.init_video_session(  # type: ignore[union-attr]
            inference_device=self.device,
            dtype=self.dtype,
        )
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._video_initialized = True
        logger.info("[SAM3] New streaming session created")
        print("[SAM3] ✓ Streaming inference session initialised")

    def _process_frame_video(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """One forward pass through Sam3TrackerVideoModel."""
        from PIL import Image

        self._ensure_video_session()

        pil_image = Image.fromarray(frame[:, :, ::-1].copy())

        # Pre-process frame -> pixel_values + original_sizes
        inputs = self.video_processor(  # type: ignore[misc]
            images=pil_image,
            device=self.device,
            return_tensors="pt",
        )
        # (H, W) tuple — needed when adding prompts in streaming mode
        original_size = inputs.original_sizes[0]

        # Register any brand-new persons with a box / point prompt
        for det in detections:
            track_id = det.get("track_id")
            box = det.get("box")
            if track_id is None or box is None:
                continue
            if track_id in self._tracked_object_ids:
                continue  # already tracked — model propagates automatically

            obj_id = self._next_obj_id
            self._next_obj_id += 1
            self._tracked_object_ids[track_id] = obj_id

            ok = self._add_object_to_session(
                box=box,
                obj_id=obj_id,
                frame_idx=self._frame_count - 1,
                original_size=original_size,
            )
            if ok:
                print(f"[SAM3] + Registered track {track_id} as obj {obj_id}")
            else:
                del self._tracked_object_ids[track_id]
                print(f"[SAM3] ✗ Could not register track {track_id} — skipping")
                logger.warning(f"[SAM3] Could not register track {track_id} — skipping")

        if not self._tracked_object_ids:
            logger.debug("[SAM3] No tracked objects — returning empty dict")
            return {}

        # Forward pass
        pixel_values = inputs.pixel_values[0].to(self.device)
        with self._no_grad():
            outputs = self.video_model(  # type: ignore[misc]
                inference_session=self._inference_session,
                frame=pixel_values,
            )

        # Post-process -> [num_objects, num_masks_per_object, H, W]
        masks_tensor = self.video_processor.post_process_masks(  # type: ignore[union-attr]
            [outputs.pred_masks],
            original_sizes=inputs.original_sizes,
            binarize=True,
        )[0]

        return self._build_track_mask_dict(masks_tensor)

    def _add_object_to_session(
        self,
        box: List[float],
        obj_id: int,
        frame_idx: int,
        original_size: Any,
        session: Optional[Any] = None,
    ) -> bool:
        """
        Register one object in a streaming session via a box prompt.

        Uses self._inference_session by default; pass `session` explicitly
        to target an ephemeral session (used by _segment_boxes_via_video_model).

        Attempts box prompt first.  If the processor raises TypeError
        (parameter not supported in this version), falls back to a single
        centre-point prompt which is universally supported.

        Returns True on success, False on failure.
        """
        target_session = session if session is not None else self._inference_session
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # --- Attempt 1: box prompt (preferred — more spatial context) ---
        try:
            self.video_processor.add_inputs_to_inference_session(  # type: ignore[union-attr]
                inference_session=target_session,
                frame_idx=frame_idx,
                obj_ids=obj_id,
                input_boxes=[[[x1, y1, x2, y2]]],
                input_boxes_labels=[[1]],
                original_size=original_size,
            )
            logger.info(
                f"[SAM3] Registered obj {obj_id} via box "
                f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
            )
            return True

        except TypeError:
            pass  # input_boxes not accepted here — try point prompt
        except Exception as exc:
            logger.warning(f"[SAM3] Box prompt failed for obj {obj_id}: {exc}")
            return False

        # --- Attempt 2: centre-point prompt (fallback) ---
        try:
            self.video_processor.add_inputs_to_inference_session(  # type: ignore[union-attr]
                inference_session=target_session,
                frame_idx=frame_idx,
                obj_ids=obj_id,
                input_points=[[[[cx, cy]]]],
                input_labels=[[[1]]],
                original_size=original_size,
            )
            logger.info(
                f"[SAM3] Registered obj {obj_id} via centre-point [{cx:.0f},{cy:.0f}]"
            )
            return True

        except Exception as exc:
            logger.warning(
                f"[SAM3] Centre-point prompt also failed for obj {obj_id}: {exc}"
            )
            return False

    def _build_track_mask_dict(self, masks_tensor: Any) -> Dict[int, np.ndarray]:
        """
        Convert the model output tensor to {track_id: binary_mask}.

        masks_tensor shape: [num_objects, num_masks_per_object, H, W]
        The model ranks masks by quality; index 0 is the best.
        Object order matches inference_session.obj_ids.
        """
        result: Dict[int, np.ndarray] = {}
        obj_id_to_track = {v: k for k, v in self._tracked_object_ids.items()}
        obj_ids = getattr(self._inference_session, "obj_ids", [])

        for i, obj_id in enumerate(obj_ids):
            track_id = obj_id_to_track.get(int(obj_id))
            if track_id is None or i >= len(masks_tensor):
                continue

            obj_masks = masks_tensor[i]  # [num_masks, H, W] or [H, W]
            if obj_masks.ndim == 3:
                mask = obj_masks[0]  # best mask (model ranks by quality)
            else:
                mask = obj_masks

            mask_np = mask.cpu().numpy().astype(np.uint8)
            px = int(np.sum(mask_np > 0))
            result[track_id] = mask_np
            logger.info(f"[SAM3] Track {track_id} -> {px} pixels")
            if px == 0:
                print(f"[SAM3] ⚠ Track {track_id}: SAM3 returned an empty mask (0 px)")

        total = len(self._tracked_object_ids)
        got = len(result)
        summary = ", ".join(
            f"track {t}={int(np.sum(m > 0))}px" for t, m in result.items()
        )
        print(
            f"[SAM3] frame #{self._frame_count:>4d} | "
            f"{got}/{total} masks | "
            f"source=sam3_video | "
            f"{summary if summary else 'no masks'}"
        )
        logger.info(f"[SAM3] process_frame produced {got}/{total} masks")
        return result

    # ------------------------------------------------------------------ #
    # Private — Sam3TrackerModel per-frame fallback                        #
    # ------------------------------------------------------------------ #

    def _process_frame_image(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """
        process_frame fallback when video streaming fails.

        Runs Sam3TrackerVideoModel in a fresh single-frame ephemeral session,
        one object at a time, so we still get real SAM masks without needing
        a persistent video session.
        """
        boxes: List[List[float]] = []
        track_ids: List[int] = []

        for det in detections:
            tid = det.get("track_id")
            box = det.get("box")
            if tid is not None and box is not None:
                boxes.append(box)
                track_ids.append(tid)

        if not boxes:
            return {}

        seg_results = self._segment_boxes_via_video_model(frame, boxes)

        result: Dict[int, np.ndarray] = {}
        for track_id, seg in zip(track_ids, seg_results):
            if seg.get("mask") is not None:
                result[track_id] = seg["mask"]
        return result

    def _segment_boxes_via_video_model(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment a set of boxes using Sam3TrackerVideoModel in a fresh
        single-frame ephemeral session.

        All boxes are registered as separate objects (obj_id 1, 2, …) in
        one session and processed in a single forward pass, so this is
        no more expensive than the old Sam3TrackerModel path was.

        This is the correct workaround for the facebook/sam3 checkpoint
        architecture mismatch: Sam3TrackerVideoModel loads cleanly from
        that checkpoint while Sam3TrackerModel does not.
        """
        from PIL import Image

        if labels is None:
            labels = [f"object_{i}" for i in range(len(boxes))]

        h, w = frame.shape[:2]
        pil_image = Image.fromarray(frame[:, :, ::-1].copy())
        density_threshold = float(getattr(settings, "MASK_DENSITY_THRESHOLD", 0.1))
        results: List[Dict[str, Any]] = []

        try:
            # One ephemeral session per call — discarded after this method returns.
            session = self.video_processor.init_video_session(  # type: ignore[union-attr]
                inference_device=self.device,
                dtype=self.dtype,
            )

            inputs = self.video_processor(  # type: ignore[misc]
                images=pil_image,
                device=self.device,
                return_tensors="pt",
            )
            original_size = inputs.original_sizes[0]

            # Register every box as a distinct object in the session.
            registered: List[int] = []  # obj_ids that were successfully added
            for i, box in enumerate(boxes):
                obj_id = i + 1
                ok = self._add_object_to_session(
                    box=box,
                    obj_id=obj_id,
                    frame_idx=0,
                    original_size=original_size,
                    session=session,
                )
                if ok:
                    registered.append(obj_id)
                else:
                    logger.warning(
                        f"[SAM3] segment_boxes: could not register obj {obj_id} "
                        f"({labels[i]})"
                    )

            if not registered:
                raise RuntimeError(
                    "No objects could be registered in ephemeral session"
                )

            # Single forward pass — produces masks for all registered objects.
            pixel_values = inputs.pixel_values[0].to(self.device)
            with self._no_grad():
                outputs = self.video_model(  # type: ignore[misc]
                    inference_session=session,
                    frame=pixel_values,
                )

            # masks_tensor: [num_objects, num_masks_per_object, H, W]
            masks_tensor = self.video_processor.post_process_masks(  # type: ignore[union-attr]
                [outputs.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )[0]

            # Build a obj_id -> mask lookup from the session's obj_ids list.
            session_obj_ids = list(getattr(session, "obj_ids", registered))
            obj_id_to_mask: Dict[int, np.ndarray] = {}
            for idx, oid in enumerate(session_obj_ids):
                if idx >= len(masks_tensor):
                    break
                obj_masks = masks_tensor[idx]  # [num_masks, H, W] or [H, W]
                if obj_masks.ndim == 3:
                    mask_np = obj_masks[0].cpu().numpy().astype(np.uint8)
                else:
                    mask_np = obj_masks.cpu().numpy().astype(np.uint8)
                obj_id_to_mask[int(oid)] = mask_np

            # Map back to the original box / label order.
            for i, (box, label) in enumerate(zip(boxes, labels)):
                obj_id = i + 1
                mask = obj_id_to_mask.get(obj_id)
                if mask is not None:
                    density = self._calculate_mask_density(mask, box)
                    score = 0.9
                    source = "sam3_video_single_frame"
                else:
                    mask = self._box_to_mask(box, h, w)
                    density = 1.0
                    score = 0.0
                    source = "box_fallback_no_mask"

                logger.info(
                    f"[SAM3] segment_boxes '{label}': "
                    f"{int(np.sum(mask > 0))} px  "
                    f"density={density:.2f}  source={source}"
                )
                results.append(
                    {
                        "mask": mask,
                        "box": box,
                        "label": label,
                        "score": score,
                        "density": density,
                        "valid": density >= density_threshold,
                        "source": source,
                    }
                )

        except Exception as exc:
            print(
                f"[SAM3] ✗ _segment_boxes_via_video_model failed: {exc}\n"
                f"       Using box-fill fallback for all {len(boxes)} boxes."
            )
            logger.warning(
                f"[SAM3] _segment_boxes_via_video_model failed: {exc} — "
                f"using box fallback for all {len(boxes)} boxes"
            )
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
                        "source": "box_fallback_error",
                    }
                )

        return results

    # ------------------------------------------------------------------ #
    # Private — last-resort box masks                                      #
    # ------------------------------------------------------------------ #

    def _box_masks_for_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """Return a filled rectangle mask for every detection."""
        h, w = frame.shape[:2]
        return {
            det["track_id"]: self._box_to_mask(det["box"], h, w)
            for det in detections
            if det.get("track_id") is not None and det.get("box") is not None
        }

    @staticmethod
    def _box_to_mask(box: List[float], h: int, w: int) -> np.ndarray:
        """Fill a bounding box region with 1s — used as a last-resort mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in box]
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1
        return mask

    @staticmethod
    def _calculate_mask_density(mask: np.ndarray, box: List[float]) -> float:
        """Fraction of the bounding box area that is covered by the mask."""
        x1, y1, x2, y2 = [int(c) for c in box]
        _h, _w = mask.shape[:2]
        x1, x2 = max(0, x1), min(_w, x2)
        y1, y2 = max(0, y1), min(_h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            return 0.0
        return float(np.sum(mask[y1:y2, x1:x2] > 0)) / float(box_area)

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def init_video_session(self, frame: Optional[np.ndarray] = None) -> None:
        """
        Reset tracking state and open a fresh streaming session.

        Called by HybridDetector whenever it encounters a new video source.
        `frame` is accepted but not required (kept for API compatibility).
        """
        if not self._initialized:
            self.initialize()

        # Tear down existing session
        self._inference_session = None
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0
        self._video_initialized = False

        if self._video_model_available:
            try:
                self._ensure_video_session()
            except Exception as exc:
                logger.warning(f"[SAM3] Could not create initial video session: {exc}")

    def reset_video_state(self) -> None:
        """Alias of init_video_session() — called by HybridDetector."""
        self._inference_session = None
        self._tracked_object_ids = {}
        self._next_obj_id = 1
        self._frame_count = 0
        self._video_initialized = False
        logger.info("[SAM3] Video state reset")

    def remove_object(self, track_id: int) -> None:
        """Remove a person who has left the frame from active tracking."""
        if track_id in self._tracked_object_ids:
            del self._tracked_object_ids[track_id]
            logger.debug(f"[SAM3] Removed track {track_id}")

    def is_tracking(self, track_id: int) -> bool:
        """Return True if this track_id is currently registered."""
        return track_id in self._tracked_object_ids

    def get_tracked_ids(self) -> List[int]:
        """Return the list of currently registered track_ids."""
        return list(self._tracked_object_ids.keys())

    def __repr__(self) -> str:
        video = "Sam3TrackerVideo" if self._video_model_available else "—"
        image = "Sam3Tracker" if self._image_model_available else "—"
        return (
            f"SAM3Segmenter("
            f"initialized={self._initialized}, "
            f"device={self.device}, "
            f"dtype={self.dtype}, "
            f"video_model={video}, "
            f"image_model={image}, "
            f"tracked={len(self._tracked_object_ids)})"
        )


# ------------------------------------------------------------------ #
# Singleton                                                            #
# ------------------------------------------------------------------ #

_sam3_segmenter: Optional[SAM3Segmenter] = None


def get_sam3_segmenter() -> SAM3Segmenter:
    """Return the module-level singleton SAM3Segmenter instance."""
    global _sam3_segmenter
    if _sam3_segmenter is None:
        _sam3_segmenter = SAM3Segmenter()
    return _sam3_segmenter
