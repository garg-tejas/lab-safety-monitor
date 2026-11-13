"""Event-level scoring of saved intervals. Never loads models or video."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = "sentinel-event-iou-v1"


def finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def validate_events(events, durations, name):
    if not isinstance(events, list):
        raise ValueError(f"{name} must be an event list")
    seen = set()
    for index, event in enumerate(events):
        prefix = f"{name}[{index}]"
        if not isinstance(event, dict):
            raise ValueError(f"{prefix}: expected an object")
        for key in ("id", "clip", "subject", "label"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise ValueError(f"{prefix}: {key} must be a nonempty string")
        if event["id"] in seen:
            raise ValueError(f"{prefix}: duplicate event id {event['id']}")
        seen.add(event["id"])
        if event["clip"] not in durations:
            raise ValueError(f"{prefix}: clip absent from declared evaluation cohort")
        start, end = event.get("start"), event.get("end")
        if not finite_number(start) or not finite_number(end) or not 0 <= start < end <= durations[event["clip"]]:
            raise ValueError(f"{prefix}: interval must be finite seconds with 0 <= start < end <= clip duration")


def interval_iou(first, second):
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    union = max(first["end"], second["end"]) - min(first["start"], second["start"])
    return overlap / union


def score_events(reference, predictions, durations, threshold=0.5):
    """Maximum-cardinality one-to-one matching per clip, subject, and label.

    IoU-ranked deterministic augmenting paths break ties, but do not maximize
    total IoU across equally sized matchings. Subject IDs must be mapped before
    scoring. Delays therefore depend on this explicitly versioned matching rule.
    """
    if not isinstance(durations, dict) or not durations:
        raise ValueError("Declare a nonempty clip-duration cohort, including negative clips")
    if any(not isinstance(clip, str) or not clip.strip() or not finite_number(seconds) or seconds <= 0 for clip, seconds in durations.items()):
        raise ValueError("Clip names must be nonempty and durations positive finite seconds")
    cohort_seconds = sum(durations.values())
    if not finite_number(cohort_seconds):
        raise ValueError("Total cohort duration must be finite")
    if not finite_number(threshold) or not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    validate_events(reference, durations, "reference")
    validate_events(predictions, durations, "predictions")
    references = sorted(reference, key=lambda event: event["id"])
    predicted = sorted(predictions, key=lambda event: event["id"])
    adjacency = {}
    for pi, event in enumerate(predicted):
        choices = []
        for ri, truth in enumerate(references):
            if any(event[key] != truth[key] for key in ("clip", "subject", "label")):
                continue
            iou = interval_iou(event, truth)
            if iou >= threshold:
                choices.append((ri, iou))
        adjacency[pi] = [ri for ri, _ in sorted(choices, key=lambda pair: (-pair[1], references[pair[0]]["id"]))]

    owners = {}

    def assign(pi, visited):
        for ri in adjacency[pi]:
            if ri in visited:
                continue
            visited.add(ri)
            if ri not in owners or assign(owners[ri], visited):
                owners[ri] = pi
                return True
        return False

    for pi in range(len(predicted)):
        assign(pi, set())
    matched_predictions = set(owners.values())
    pairs = [
        {
            "reference_id": references[ri]["id"],
            "prediction_id": predicted[pi]["id"],
            "iou": interval_iou(references[ri], predicted[pi]),
            "onset_difference_seconds": predicted[pi]["start"] - references[ri]["start"],
        }
        for ri, pi in sorted(owners.items())
    ]
    tp = len(pairs)
    fp, fn = len(predicted) - tp, len(references) - tp
    delays = [pair["onset_difference_seconds"] for pair in pairs]
    return {
        "protocol": PROTOCOL,
        "iou_threshold": threshold,
        "cohort_seconds": cohort_seconds,
        "clip_count": len(durations),
        "reference_events": len(references),
        "predicted_events": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / len(predicted) if predicted else None,
        "recall": tp / len(references) if references else None,
        "false_alerts_per_minute": fp / (cohort_seconds / 60),
        "mean_signed_onset_difference_seconds": sum(delays) / len(delays) if delays else None,
        "matches": pairs,
        "unmatched_prediction_ids": [event["id"] for i, event in enumerate(predicted) if i not in matched_predictions],
        "unmatched_reference_ids": [event["id"] for i, event in enumerate(references) if i not in owners],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with durations, reference, predictions, provenance")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Input must be a JSON object")
        provenance = payload.get("provenance", {})
        required = ("cohort_id", "annotations_version", "prediction_run_id", "model_revision", "pipeline_config", "identity_mapping")
        if not isinstance(provenance, dict) or any(not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in required):
            raise ValueError("Provenance requires nonempty strings: " + ", ".join(required))
        report = score_events(payload["reference"], payload["predictions"], payload["durations"], args.iou)
        report.update({
            "provenance": provenance,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Scores only supplied saved intervals; no inference, annotation verification, or measured throughput.",
        })
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
