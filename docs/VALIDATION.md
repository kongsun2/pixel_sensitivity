# Validation record — 2026-11-04

- Reference implementation extracted from the supplied manual, then checked in the actual environment.
- Final pytest run: **38 passed**. One external anyio plugin deprecation warning; no failures or skips. Log: `logs/tests_final.log`.
- Real GPU CLI smoke: a 3-image batch including a partial last batch produced exactly the same predictions for clean and zero-alpha faults in every cell.
- Two-GPU smoke: all four disjoint grid shards were generated; predictions from both RTX 4090 GPUs exactly matched the clean zero-alpha reference. Logs: `logs/software_smoke_gpu0.log`, `logs/software_smoke_gpu1.log`.
- Bulk annotation coverage: 2,300 unique image records, 3,563 boxes. 1,657 existing individually downloaded YOLO labels cross-checked against the same-revision bulk annotations.
- Source audit: eight unverified empty-label images excluded; no exact decoded image duplicates; all ten classes represented in every split.
- Training: successful early stopping after 31 epochs; best checkpoint selected using validation data.
- Real experiment: **529 complete inference shards**, **187,266 image evaluations**; all shards checked by source/data/model/config identity and SHA256 during analysis.
- Final analysis contracts: **176 condition/grid summary rows**, **528 condition/repeat/grid rows**, **186,912 clean/fault paired-image rows**. No partial paired CSV remains.
- Generated 15 PNG figures and a Korean results report; combined sensitivity map inspected visually.
- Code review addressed source hashes in cache identity and exclusion of unverified empty-label images. Follow-up review found no additional important race or correctness issues in disjoint GPU workers.

These checks validate the pipeline and this recorded experiment. They do not validate a physical sensor model or generalization across training seeds, cameras, coils, or datasets. Pointwise intervals are not a multiple-comparison-corrected test of differences among locations.
