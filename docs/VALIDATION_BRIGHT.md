# Bright replication validation — 2026-11-05

- Final automated tests: **40 passed**, one external anyio deprecation warning.
- Official PCB-IND archive: Zenodo record 19723114, MD5 `1325f8dffeb73bef4db34fbd22b31ec5`; official COCO labels used after identifying YOLO boundary rounding. COCO dimensions match all 4789 images. 127 empty/unverified labels excluded.
- NEU official archive: 1800 images. Conflicting identical-image pair excluded, three duplicate boxes removed, difficult objects explicitly retained, 1-based-inclusive VOC coordinates tested.
- Independent original YOLOv8n COCO initialization, seed11: NEU early stopping53, PCB42 epochs. Original train.py, run.py, core.py, analyze.py and coco_metrics.py unchanged from the GC10 implementation.
- NEU: 241 complete shards, 65070 image evaluations, 80 grid summaries, 240 condition/repeat/grid rows, 64800 paired-image rows.
- PCB: 241 complete shards, 113029 image evaluations, 80 grid summaries, 240 condition/repeat/grid rows, 112560 paired-image rows.
- Analysis validated prediction shard SHA256, run identity and unchanged GT. Final source and model hashes checked against recorded run metadata. No partial analysis CSV remains.
- Three-dataset report asserts the five bright-condition definitions and all remaining experiment settings are identical across datasets. GC10 results are read without modification.
- Review identified PCB numeric IDs crossing official splits despite side-prefix separation: 683 numeric IDs, 124 cross-split, 63 train/test overlap. Official split preserved with explicit board-independence and bootstrap caveats; no independent-board generalization claim.
- Per-dataset Korean reports, common-scale comparative map and numeric CSVs generated; comparative map inspected visually.

This is a single-training-seed, native-image-space fixed-count comparison. Native resolutions and resize factors differ; physical sensor failures, board-independent generalization and multi-seed reproducibility are not established.
