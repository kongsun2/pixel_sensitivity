# 최종 YOLOv8n 가중치

각 파일은 해당 학습의 검증 성능 기준으로 선택된 `best.pt`를 원본 그대로 복사한 것입니다. 현행 주 실험은 GC10·NEU·숫자 ID 그룹 분리 PCB의 시드 11·22·33 총 9개 모델이며, `pcb_ind_yolov8n_s11`은 기존 공식 분할 비교용입니다.

| 학습 이름 | 가중치 | 용량 (bytes) |
| --- | --- | ---: |
| gc10_yolov8n_s11 | [best.pt](gc10_yolov8n_s11/best.pt) | 6224938 |
| neu_det_yolov8n_s11 | [best.pt](neu_det_yolov8n_s11/best.pt) | 6252586 |
| pcb_ind_yolov8n_s11 | [best.pt](pcb_ind_yolov8n_s11/best.pt) | 6251946 |
| validation_20261111_gc10_det_s22 | [best.pt](validation_20261111_gc10_det_s22/best.pt) | 6226026 |
| validation_20261111_gc10_det_s33 | [best.pt](validation_20261111_gc10_det_s33/best.pt) | 6228842 |
| validation_20261111_neu_det_s22 | [best.pt](validation_20261111_neu_det_s22/best.pt) | 6255530 |
| validation_20261111_neu_det_s33 | [best.pt](validation_20261111_neu_det_s33/best.pt) | 6258602 |
| validation_20261111_pcb_ind_board_s11 | [best.pt](validation_20261111_pcb_ind_board_s11/best.pt) | 6258666 |
| validation_20261111_pcb_ind_board_s22 | [best.pt](validation_20261111_pcb_ind_board_s22/best.pt) | 6256746 |
| validation_20261111_pcb_ind_board_s33 | [best.pt](validation_20261111_pcb_ind_board_s33/best.pt) | 6254826 |

학습 설정과 곡선은 `../reports/training_summaries/`에서, 원본 위치와 SHA256은 `../reports/ARTIFACT_MANIFEST.json`에서 확인할 수 있습니다. 데이터셋은 포함하지 않습니다. 실행 스크립트의 `--weights`에 이 디렉터리의 해당 `best.pt` 경로를 지정해 추론할 수 있습니다.
