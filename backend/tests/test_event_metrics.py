"""Small synthetic interval fixtures only; no video, model, DB or API execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval.event_metrics import score_events


def event(identity, start=0, end=10, **changes):
    return {"id": identity, "clip": "clip", "subject": "person", "label": "missing_mask", "start": start, "end": end, **changes}


class EventMetricTests(unittest.TestCase):
    def test_duplicate_alerts_are_false_positives_not_extra_matches(self):
        report = score_events([event("truth")], [event("a"), event("b")], {"clip": 60})
        self.assertEqual((report["true_positives"], report["false_positives"]), (1, 1))
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["false_alerts_per_minute"], 1)

    def test_matching_maximizes_event_count_instead_of_greedy_best_edge(self):
        reference = [event("truth-a", 0, 10), event("truth-b", 5, 15)]
        predictions = [event("a", 0, 15), event("b", 0, 6)]
        report = score_events(reference, predictions, {"clip": 60})
        self.assertEqual(report["true_positives"], 2)

    def test_identity_label_and_clip_are_not_silently_merged(self):
        report = score_events([event("truth")], [event("p", subject="other")], {"clip": 60})
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["false_positives"], 1)

    def test_negative_clips_contribute_to_false_alert_denominator(self):
        report = score_events([], [event("p")], {"clip": 60, "negative": 60})
        self.assertIsNone(report["recall"])
        self.assertEqual(report["false_alerts_per_minute"], 0.5)

    def test_no_predictions_is_undefined_precision_and_zero_recall(self):
        report = score_events([event("truth")], [], {"clip": 60})
        self.assertIsNone(report["precision"])
        self.assertEqual(report["recall"], 0)

    def test_onset_difference_is_signed_and_matches_are_order_independent(self):
        reference = [event("a", 1, 11), event("b", 20, 30)]
        predictions = [event("p", 0, 11), event("q", 21, 30)]
        first = score_events(reference, predictions, {"clip": 60})
        second = score_events(reference[::-1], predictions[::-1], {"clip": 60})
        self.assertEqual(first, second)
        self.assertEqual(first["mean_signed_onset_difference_seconds"], 0)

    def test_invalid_intervals_and_duplicate_ids_fail(self):
        for predictions in [[event("p", 3, 3)], [event("p", end=61)], [event("p", start=float("nan"))], [event("p"), event("p")], [event("p", clip="missing")]]:
            with self.assertRaises(ValueError):
                score_events([], predictions, {"clip": 60})

    def test_invalid_threshold_and_duration_fail(self):
        for threshold in (0, 1.1, True, float("nan")):
            with self.assertRaises(ValueError):
                score_events([], [], {"clip": 60}, threshold)
        for durations in ({}, {"clip": 0}, {"clip": True}, {"clip": 10 ** 500}, {"a": 1e308, "b": 1e308}):
            with self.assertRaises(ValueError):
                score_events([], [], durations)

    def test_cli_requires_provenance_and_never_overwrites_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "events.json", Path(directory) / "report.json"
            payload = {"reference": [], "predictions": [], "durations": {"clip": 60}}
            source.write_text(json.dumps(payload))
            command = [sys.executable, "-m", "eval.event_metrics", str(source), "--output", str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            payload["provenance"] = {key: "synthetic-fixture" for key in ("cohort_id", "annotations_version", "prediction_run_id", "model_revision", "pipeline_config", "identity_mapping")}
            source.write_text(json.dumps(payload))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            before = output.read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
