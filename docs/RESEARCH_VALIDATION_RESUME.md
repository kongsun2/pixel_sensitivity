# Remaining research validation — live run and resume notes

User authorized all remaining validations and requests completed results, not just a plan. Work started2026-11-11. Dataset downloads not needed.

## Stable scope

- Existing ten-mask caches: distribution/class/size/GT-hit descriptive analysis and exploratory joint group×mask simultaneous spatial bands completed.
- New PCB canonical: `data/canonical/pcb_ind_board`,3295train/728val/639test,683numeric groups, no crosssplit numericIDs, all8classes. Old official cohort remains separate.
- Seven new models trained and complete in `reports/research_validation_2026-11-11/training_state.json`: GC10/NEU seeds22,33 plusPCBboard seeds11,22,33. OriginalGC10/NEU seed11 reused; mainstudy9models.
-27jobs: main3datasets×(seed11 sevenconditions +seed22/33 bright25only). Conditions: bright25,dark25,stuck128/25,bright1,bright9,bright25alpha.5,bright5x5cluster25. Grid8x8,10masks, sameFP32batch16/preprocessing/thresholds/COCO evaluator.
- Mask namespace `pcb_ind` explicitly reused for new `pcb_ind_board`, preserving old native300px maskgeometry.
- Totalvalid main image-evaluation results7286247,641shards perjob. Parallel logical partitions8, max2processes perGPU,2RTX4090.

## Current master

- PID recorded at `logs/research_validation/master_pid.json`. Actual entry `scripts/run_research_validation_compiled.py`.
- Log `logs/research_validation/master_compiled.log`.
- Progress `.venv/bin/python scripts/status_research_validation.py`.
- Current manifest/state: `reports/research_validation_2026-11-11/evaluation_manifest.json` and `evaluation_state.json`.
- Runs: `runs/research_validation_compiled_20261111/{dataset}_s{seed}__{condition}/`.
- DO NOT start a duplicate master while PID is alive. Detached process should survive chat interruptions. If absent, execute the same compiled master; complete outputs and numerical-control fingerprints are checked before reuse. Partially written shards are redone.
- Source `run_compiled.py`, `safe_compiled_nms.py`, `run_fast.py`, original `run.py/core.py/analyze.py/train.py` are FROZEN while execution runs; changing them invalidates identity.

## Execution optimization/provenance

- Initial eagerFP32 master and its partialresults retained under `runs/research_validation_20261111`, archived state/manifest `*_eager_partial.json`; not main27job counts.
- CUDAgraph prototype did not help and is NOT integrated.
- Guardedcompiled NMS does preserve original greedy argsort/tieorder; near-IoU/invalid/large-candidate cases fall back to originalTorchNMS. Confidence is not altered. This improved execution without changing research conditions.
- Global numerical certificate `compiled_backend_validation.json`:3original models×130fixedvalidationimages×8controls, exact predictiondict/perimage matching, APdifference0.
- Every main checkpoint also gets17validationimages×8control exact checks; paths inmanifest `model_control_path`.
- First completeGC10bright25 reproduced all640previous case metrics exactly; master checksGC10/NEU reference parity and records maxdifference.
- Initialsoftware tests passed; reporter/NMS tests also exist. Final run all tests and verify report/counts when done.

## Final automatic steps

After27inference/analysisjobs, master re-runs distribution andpositiondiagnostics on correctedPCBseed11bright25, then invokes `scripts/report_research_validation.py`. Final report `reports/research_validation_2026-11-11/summary/REPORT_KO.md`, completion marker `reports/research_validation_2026-11-11/COMPLETE.json`.

Before final answer: verify27complete states, matrix9models/7conditions asdeclared, outputCSVcounts, source/model hashes, figures visually. Summarize observed condition effects, seed robustness, correctedPCB split result, and conditional spatial/exposure conclusions. No physical RAW/ISP/DPC or intrinsic sensor-position causality claim. CorrectedPCB testcohort changes, so old-vs-new performance differences cannot be attributed only to leakage removal.
