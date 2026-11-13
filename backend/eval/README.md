# Offline event evaluation

This standard-library evaluator scores saved events, not video or detections. It loads no ML, API, database, or application modules. No real benchmark has been run for this change.

Use a fixed, annotated clip cohort, including clips without violations. Export one event per violation label with `id`, `clip`, `subject`, `label`, `start`, and `end` (seconds from clip start). Split multi-label events into separate label records. Resolve ongoing events at the clip boundary. Map predicted subject/track IDs to annotated subject IDs explicitly; the evaluator does not infer identity correspondences or perform tracking evaluation.

Input is one JSON object with:

- `durations`: object mapping every cohort clip ID to its duration in seconds.
- `reference` and `predictions`: lists of event objects using the fields above.
- `provenance`: nonempty strings for `cohort_id`, `annotations_version`, `prediction_run_id`, `model_revision`, `pipeline_config`, and `identity_mapping`. Use immutable versions/hashes where available, not a mutable `latest` tag.

Run from `backend/` when saved records are ready:

```sh
python -m eval.event_metrics saved-events.json --output new-report.json --iou 0.5
```

Reports contain the input/evaluator hashes, provenance, TP/FP/FN, event precision/recall, false alerts per minute of declared cohort, signed onset differences, matches and unmatched IDs. Output files must be new. Missing/invalid fields fail rather than producing apparently valid scores. Empty-denominator precision/recall are null, not zero. The evaluator does not verify annotation quality or provenance truthfulness.

Matching is maximum-cardinality one-to-one at the selected interval IoU, within exactly matching clip/subject/label keys. Deterministic IoU-ranked augmenting paths select among matches; they do not maximize total IoU among all maximum-cardinality assignments. Delay interpretation depends on that tie-breaking. Early predicted onsets produce negative signed differences. Duplicate prediction events compete for a single reference and therefore count as extra false positives. Choose thresholds and annotation rules before inspecting the held-out results.

For a later raw/EMA/EMA+hysteresis ablation, keep the cohort, annotations, label taxonomy, ID mapping policy and IoU threshold fixed. Preserve failures and negative clips. This tool does not claim person-independent generalization, model FPS, processing latency, frame-drop rate, or causal benefit from filtering. Those need separately recorded data. Very large overlapping event sets require an optimized matcher; this implementation is intended for a small labeled clip study.

Offline fixture tests:

```sh
python -m unittest discover -s tests -p test_event_metrics.py -q
```
