# 결과 파일 안내

## 현행 주 결과

- [연구 검증 종합 보고서](research_validation_2026-11-11/summary/REPORT_KO.md): 3개 데이터셋, 총 27개 모델/조건 평가.
- [원본 성능](research_validation_2026-11-11/summary/clean_baselines.csv): 데이터셋·학습 시드별 손상 미주입 기준값.
- [조건별 평균](research_validation_2026-11-11/summary/condition_means.csv): 손상 유형·화소 수·강도·배치 비교.
- [패턴별 평균](research_validation_2026-11-11/summary/pattern_means.csv): 각 패턴의 64개 위치 평균.
- [학습 시드별 결과](research_validation_2026-11-11/summary/bright25_seed_results.csv), [시드 평균](research_validation_2026-11-11/summary/bright25_seed_summary.csv).
- [위치 지도](research_validation_2026-11-11/summary/training_seed_spatial_maps.png), [조건 비교 그림](research_validation_2026-11-11/summary/rq3_conditions_recall_drop.png).
- [현행 PCB 결함 분포 분석](research_validation_2026-11-11/corrected_pcb_distribution/REPORT_KO.md), [위치 공동 구간](research_validation_2026-11-11/corrected_pcb_positions/REPORT_KO.md).
- [PCB 분할 검증](research_validation_2026-11-11/pcb_split/REPORT_KO.md): 숫자 ID는 실제 물리 보드가 확인된 값이 아닌 대리 그룹.

시드 11에서 7개 조건을 평가했고 시드 22·33에서는 밝은 25화소 조건만 반복했습니다. 각 손상 조건은 8×8 위치 × 10개 패턴입니다. 10개는 손상 배치 반복이며 10가지 손상 유형을 의미하지 않습니다. 현행 PCB 결과는 `pcb_ind_board`이며 `pcb_ind`는 과거 공식 분할입니다. 감소량은 원본−손상 후이므로 양수는 성능 저하입니다.

## 보고서

- [재정리 중간보고서 Word](progress_report_reorganized_2026-11-12/pilot_progress_report_reorganized.docx), [PDF](progress_report_reorganized_2026-11-12/pilot_progress_report_reorganized.pdf), [HTML](progress_report_reorganized_2026-11-12/report.html).
- [모델 설명 상세 Word 초안](interim_model_2026-11-13/interim_model_yolov8n.docx): YOLOv8n의 구조와 사용 방법. 이후 대화에서 정리한 축약 본문과는 별도의 상세 초안입니다.

보고서 형식 참고용 외부 문서 자체와 그 추출 내용은 포함하지 않습니다. PDF와 Word의 본문 내용은 같지만 페이지 배치는 동일 렌더링 결과가 아닙니다.

## 이전 실험 및 감사 자료

- `bright_replication_2026-11-05/`: 초기 4×4 밝은 화소 비교.
- `bright25_grid8_2026-11-10/`: 8×8·3패턴 결과 및 실제 주입 위치.
- `bright25_grid8_pattern10_2026-11-10/`: 기존 모델의 8×8·10패턴 결과.
- `research_validation_2026-11-11/`: 완료 표식, 실행 목록, 대조 검증, 분포·위치 통계, 종합 결과.
- `progress_report_reorganized_2026-11-12/`: 보고서 작성 근거와 추가 감사. GC10 그룹은 물리 제품 독립성을 보장하지 않으며 추가 파일명 그룹 감사도 여기에 보존합니다.

## 보관된 실행 요약

- `run_summaries/research_validation_compiled_20261111/`: 현행 27개 평가 작업의 위치·조건별 지표, 원본 지표, 지도, 실행 메타데이터.
- `run_summaries/evaluation/`: 이전 평가의 작은 분석 산출물과 메타데이터.
- `training_summaries/`: 학습별 `args.yaml`, `results.csv`, 학습·평가 곡선 등. 가중치는 제외합니다.
- [최종 학습 가중치](../models/README.md): 현행 9개 모델과 기존 PCB 공식 분할 비교 모델의 `best.pt`.
- [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json): 복사한 파일의 원본 상대 경로·SHA256·크기, 제외 산출물의 종류와 규모.

원본 데이터, 중간 가중치, 이미지별 `paired_images.csv`, 대용량 예측 캐시는 로컬에 유지하며 Git에는 포함하지 않습니다. 최종 `best.pt`는 `models/`에 포함하므로 데이터가 준비되어 있으면 재학습 없이 추론할 수 있습니다. 예측 캐시가 없는 체크아웃만으로 해당 캐시를 요구하는 보고서 생성기를 재실행할 수는 없습니다. 저장소의 기존 결과표와 지도를 바로 열어 확인할 수 있고, 완전 재현에는 데이터 준비·추론을 다시 수행하거나 원래 로컬 산출물이 필요합니다.
