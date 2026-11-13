# Ten-pattern validation — 2026-11-10

- 51 automated tests passed, including11 tests for cache/model/data/config compatibility. One external anyio deprecation warning.
- Ten unique dispersed placement masks per dataset, 25sites each, fit every8x8cell. Same models, test IDs, preprocessing, injection rules, thresholds, source and environment verified before reusingP01-P03.
- P01-P03 prediction files copied with source SHA256 and original run-hash provenance; prior files unchanged. P04-P10 inferred on twoRTX4090GPUs. Execution resumed from completed shards after interruption.
- Each dataset has641complete shards (193verified reused,448new),640per-location/pattern metric rows and64location summaries.
- Final unchanged-source/model hash checks passed. First3patterns reproduce previous TP/FP/FN, Recall, Precision,F1 and COCO AP values to1e-12.
- Per-pattern means include all64locations. All10pattern means equal the dataset grand mean computed directly from640metric rows.
- Thirty pattern/dataset rows and3dataset grand-mean rows exported, with all4requested detection metrics plus auxiliary values.
- Total700613image evaluation results, including489664new results and210949verified existing results. Paired-image rows699520. No partial analysis CSV remains.
- Code review found no important issue in compatibility contract or averaging/units. Report and comparison/pattern figures generated and inspected.

These are ten randomized dispersed placements rather than ten structural fault families. Pattern numbers are local to each dataset, not matched spatial shapes across resolutions. Existing single-training-seed, ROI, native-resolution and split-independence limitations remain.
