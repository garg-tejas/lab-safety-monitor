"""CPU-only contract tests with all model/runtime dependencies stubbed.

Run: python -m unittest discover -s tests -v
No torch/transformers/InsightFace imports, downloads, inference or database calls.
"""
import ast
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, APP / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def isolated_function(path, name, namespace):
    """Load one real route/helper body without application startup imports."""
    tree = ast.parse((APP / path).read_text())
    node = next(item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP / path), "exec"), namespace)
    return namespace[name]


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            PPE_PROMPTS=["lab coat"], DETECTION_CONFIDENCE_THRESHOLD=0.5,
            REQUIRED_PPE=[], USE_MOCK_FACE=False, USE_MOCK_DETECTOR=False,
            DETECTOR_TYPE="hybrid", YOLOV11_MODEL_PATH=None, FRAME_SAMPLE_RATE=3,
        )
        self.fake_torch = module("torch", cuda=SimpleNamespace(is_available=lambda: False), float32="float32")
        self.patch = patch.dict(sys.modules, {
            "app": module("app"), "app.ml": module("app.ml"), "app.core": module("app.core"),
            "app.core.config": module("app.core.config", settings=self.settings),
            "torch": self.fake_torch, "cv2": module("cv2", putText=Mock(), FONT_HERSHEY_SIMPLEX=0, VideoCapture=Mock()),
            "PIL": module("PIL", Image=SimpleNamespace(fromarray=lambda frame: SimpleNamespace(convert=lambda mode: frame))),
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.error = load("app.core.model_status", "core/model_status.py").ModelUnavailableError
        self.frame = np.zeros((20, 20, 3), dtype=np.uint8)

    def sam(self, mock=False):
        return load("app.ml.sam3_detector", "ml/sam3_detector.py").SAM3Detector(mock_mode=mock)

    def test_sam_load_failure_never_falls_back_to_mock(self):
        broken = SimpleNamespace(from_pretrained=Mock(side_effect=RuntimeError("missing weights")))
        sys.modules["transformers"] = module("transformers", Sam3Model=broken, Sam3Processor=broken)
        detector = self.sam()
        detector._mock_detect = Mock(side_effect=AssertionError("fake detections forbidden"))
        with self.assertRaisesRegex(self.error, "SAM3 unavailable"):
            detector.detect(self.frame)
        self.assertFalse(detector._initialized)
        detector._mock_detect.assert_not_called()

    def test_sam_inference_error_never_falls_back(self):
        detector = self.sam()
        detector._initialized = detector._model_available = True
        detector.model = object()
        detector.processor = Mock(side_effect=RuntimeError("runtime unavailable"))
        detector._mock_detect = Mock()
        with self.assertRaisesRegex(self.error, "inference failed"):
            detector.detect(self.frame)
        detector._mock_detect.assert_not_called()

    def test_explicit_sam_mock_is_labeled_and_does_not_load_models(self):
        detector = self.sam(mock=True)
        detector.initialize()
        result = detector.detect(self.frame)
        self.assertTrue(result["simulated"])
        self.assertEqual(result["inference_mode"], "mock")
        self.assertIsNone(detector.model)
        detector.associate_ppe_to_persons(result["persons"], result["ppe_detections"], {}, [])

    def test_face_load_failure_never_invents_identity(self):
        sys.modules["insightface.app"] = module("insightface.app", FaceAnalysis=Mock(side_effect=RuntimeError("missing")))
        sys.modules["insightface"] = module("insightface")
        recognizer = load("app.ml.face_recognition", "ml/face_recognition.py").FaceRecognizer()
        recognizer._mock_detect = Mock()
        with self.assertRaisesRegex(self.error, "No identities"):
            recognizer.detect_faces(self.frame)
        recognizer._mock_detect.assert_not_called()

    def test_explicit_face_mock_is_labeled(self):
        recognizer = load("app.ml.face_recognition", "ml/face_recognition.py").FaceRecognizer(mock_mode=True)
        self.assertTrue(recognizer.detect_faces(self.frame)[0]["simulated"])

    def test_yolo_missing_weights_and_runtime_error_fail_closed(self):
        detector = load("app.ml.yolov11_detector", "ml/yolov11_detector.py").YOLOv11Detector()
        with self.assertRaisesRegex(self.error, "not configured"):
            detector.initialize()
        detector.model, detector.model_type, detector._initialized = object(), "pytorch", True
        detector._detect_pytorch = Mock(side_effect=RuntimeError("unavailable"))
        detector._mock_detect = Mock()
        with self.assertRaisesRegex(self.error, "inference failed"):
            detector.detect(self.frame)
        detector._mock_detect.assert_not_called()

    def test_mock_factory_overrides_hybrid_without_mutating_real_singleton(self):
        sam_module = load("app.ml.sam3_detector", "ml/sam3_detector.py")
        sys.modules["app.ml.yolov11_detector"] = module("app.ml.yolov11_detector", YOLOv11Detector=object, get_yolov11_detector=Mock())
        factory = load("app.ml.detector_factory", "ml/detector_factory.py")
        self.settings.USE_MOCK_DETECTOR = True
        self.assertTrue(factory.get_detector().mock_mode)
        self.settings.USE_MOCK_DETECTOR = False
        self.settings.DETECTOR_TYPE = "sam3"
        self.assertFalse(factory.get_detector().mock_mode)
        self.assertIsNot(factory._mock_detector, sam_module._detector)

    def test_unknown_detector_type_is_configuration_error(self):
        load("app.ml.sam3_detector", "ml/sam3_detector.py")
        sys.modules["app.ml.yolov11_detector"] = module("app.ml.yolov11_detector", YOLOv11Detector=object, get_yolov11_detector=Mock())
        factory = load("app.ml.detector_factory", "ml/detector_factory.py")
        self.settings.DETECTOR_TYPE = "typo"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            factory.get_detector()

    def test_failed_job_initialization_records_error_before_video_or_database_access(self):
        pipeline = SimpleNamespace(initialize=Mock(side_effect=self.error("Missing local weights")))
        manager = SimpleNamespace(update_job=Mock())
        process = isolated_function("api/routes/stream.py", "process_video_task", {
            "get_pipeline": lambda: pipeline, "processing_manager": manager, "ModelUnavailableError": self.error,
        })
        asyncio.run(process("job-1", "example.mp4"))
        self.assertEqual(manager.update_job.call_args.kwargs, {"status": "failed", "error": "Missing local weights"})

    def test_mock_results_never_reach_persistence(self):
        tree = ast.parse((APP / "services/persistence.py").read_text())
        cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "PersistenceManager")
        method = next(item for item in cls.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "persist_frame_results")
        namespace = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "persistence.py", "exec"), namespace)
        result = asyncio.run(namespace[method.name](object(), {"simulated": True, "persons": [{"person_id": "fake"}]}, None))
        self.assertEqual(result["created_events"], 0)
        self.assertTrue(result["simulated"])

    def pipeline(self, detector):
        sys.modules["app.ml.detector_factory"] = module("app.ml.detector_factory", get_detector=lambda: detector)
        temporal = SimpleNamespace(clear_all=Mock())
        sys.modules["app.ml.temporal_filter"] = module("app.ml.temporal_filter", get_temporal_filter=lambda: temporal)
        sys.modules["app.ml.mask_utils"] = module("app.ml.mask_utils", draw_person_with_ppe=Mock(), draw_frame_info=Mock())
        return load("app.ml.pipeline", "ml/pipeline.py").DetectionPipeline()

    def test_pipeline_status_clearly_reports_unavailable_models(self):
        detector = SimpleNamespace(initialize=Mock(side_effect=ImportError("missing model runtime")))
        pipeline = self.pipeline(detector)
        with self.assertRaisesRegex(self.error, "pipeline unavailable"):
            pipeline.initialize()
        self.assertFalse(pipeline.status()["ready"])
        self.assertEqual(pipeline.status()["inference_mode"], "real")
        self.assertIn("missing model runtime", pipeline.status()["error"])

    def test_pipeline_mock_output_has_metadata_and_visible_watermark(self):
        detector = SimpleNamespace(
            mock_mode=True, initialize=Mock(), detect=lambda frame: {"simulated": True, "persons": []},
            associate_ppe_to_persons=lambda *args: [],
        )
        pipeline = self.pipeline(detector)
        pipeline.initialize()
        pipeline._annotate_frame = lambda frame, *args: frame
        result = pipeline.process_frame(self.frame)
        self.assertTrue(result["simulated"])
        self.assertEqual(pipeline.status()["inference_mode"], "mock")
        self.assertIn("SIMULATED", sys.modules["cv2"].putText.call_args.args[1])

    def test_model_failure_invalidates_pipeline_readiness(self):
        detector = SimpleNamespace(initialize=Mock(), detect=Mock(side_effect=self.error("model runtime failed")))
        pipeline = self.pipeline(detector)
        pipeline.initialize()
        with self.assertRaises(self.error):
            pipeline.process_frame(self.frame)
        self.assertFalse(pipeline.status()["ready"])

    def test_health_endpoint_never_loads_a_model(self):
        state = {"ready": False, "inference_mode": "real", "error": "weights unavailable"}
        health = isolated_function("main.py", "health_check", {"get_pipeline": lambda: SimpleNamespace(status=lambda: state)})
        result = asyncio.run(health())
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["inference"], state)

    def test_face_runtime_failure_never_becomes_empty_success(self):
        recognizer = load("app.ml.face_recognition", "ml/face_recognition.py").FaceRecognizer()
        recognizer.app = SimpleNamespace(get=Mock(side_effect=RuntimeError("runtime failed")))
        recognizer._initialized = True
        with self.assertRaisesRegex(self.error, "inference failed"):
            recognizer.detect_faces(self.frame)

    def test_live_worker_clears_stale_detections_after_model_failure(self):
        sys.modules["app.core.database"] = module("app.core.database", async_session=Mock())
        sys.modules["app.services"] = module("app.services")
        sys.modules["app.services.persistence"] = module("app.services.persistence", PersistenceManager=Mock())
        processor_class = load("app.ml.stream_processor", "ml/stream_processor.py").StreamProcessor
        pipeline = SimpleNamespace(process_frame=Mock(side_effect=self.error("weights unavailable")))
        processor = processor_class(pipeline)
        processor.result_cache = {"persons": [{"id": "stale"}]}
        processor.running = True
        calls = iter([(self.frame, "test")])

        def get_frame(**kwargs):
            try:
                return next(calls)
            except StopIteration:
                processor.running = False
                from queue import Empty
                raise Empty

        processor.frame_queue = SimpleNamespace(get=get_frame, task_done=Mock())
        processor._processing_worker()
        self.assertEqual(processor.result_cache["persons"], [])
        self.assertEqual(processor.result_cache["inference_mode"], "unavailable")
        sys.modules["app.core.database"].async_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
