# Bright pixel replication on PCB-IND and NEU-DET

User requests the same bright pixel experiments on the remaining proposal datasets. Reuse validated training, injection, inference and evaluation modules without changing the GC10 experiment.

- [x] Obtain official archives and verify hashes.
- [x] Audit labels, duplicate pixels/annotations and splits; document import conventions.
- [x] Independently train YOLOv8n from the same original COCO checkpoint, seed=11, max 100 epochs, patience=20.
- [x] Run clean plus 4x4 grid x 5 fixed bright conditions x 3 pattern repetitions.
- [x] Calculate same COCO AP, paired recall confidence intervals, occupancy and confidence metrics.
- [x] Generate per-dataset reports and a three-dataset bright-condition comparison with resolution/dose caveats.
