# Research validation implementation plan

Goal: Complete the user-authorized remaining analyses and experiments, with auditable comparisons and honest limits.
Architecture: Preserve existing source/caches. Analyze original ten-mask caches for exposure/statistics; prepare PCB numeric-group split; independently train7newmodels; evaluate a fixed matched-condition matrix using a numerically validated faster FP32 runner; produce one integrated Korean report.
Tech: Existing Python/Ultralytics8.3.221/PyTorch/CUDA, NumPy/Pandas/COCOeval/Matplotlib.

- [x] Existing-cache distribution analysis: class/size/hit strata and descriptive spatial residuals.
- [x] Existing-cache simultaneous position-effect bands with joint image-group and mask resampling.
- [x] PCB numeric-prefix grouped70/15/15split(seed101), zero crosssplit groups, class/label audit.
- [x] Train GC10/NEU seeds22,33 and PCBboard seeds11,22,33 from original COCO yolov8n.pt. Same100max epochs/patience20 and recipe as existing.
- [x] Validate fast FP32 inference against original backend; execution optimization only, no model/threshold tuning. Freeze chosen execution batch and record numerical differences.
- [x] For seed11 main models run baseline bright25 plus six matched additions: dark25, stuck128/25, bright1, bright9, bright25alpha0.5, brightcluster5x5count25. Grid8x8,10masks. Same sites/nested budgets for dispersed conditions.
- [x] Run bright25 for the two additional training seeds on each main dataset. All PCB main models use corrected split; old official-split result remains exploratory comparison.
- [x] Run unchanged operational/COCO evaluation and paired bootstrap, summarize conditions and seed variation.
- [x] Reapply exposure/position diagnostics to corrected PCB baseline if scripts support it.
- [x] Verify all counts, hashes and numerical comparisons, produce Korean report/CSVs/figures and scope limitations.

Interpretation: No independent-board claim from numeric ID alone. Group split changes test cohort, so old-vs-new model differences are not a causal estimate of leakage. Existing ten-mask results are exploratory. Descriptive class/size/hit adjustment does not establish intrinsic sensor-coordinate causality. Actual RAW/ISP/DPC remain outside scope.
