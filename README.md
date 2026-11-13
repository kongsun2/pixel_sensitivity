# 제조 영상 화소 손상 위치별 민감도 파이프라인

## 최신 결과와 저장소 범위

GC10-DET, NEU-DET, PCB-IND에서 YOLOv8n을 독립적으로 추가 학습하고, 원본과 화소 손상 영상의 탐지 성능을 비교한 연구 프로젝트입니다. 현재 주 결과는 8×8 위치와 10개 패턴을 사용하는 27개 모델/조건 평가이며, 학습 시드 11에서 7개 조건을, 시드 22·33에서 밝은 25화소 조건을 평가했습니다. 현행 PCB 주 결과는 숫자 ID 그룹을 분리한 `pcb_ind_board`이며, 기존 공식 분할 결과는 비교 이력입니다.

- **주 결과 보고서:** [추가 검증 결과](reports/research_validation_2026-11-11/summary/REPORT_KO.md)
- **평균 결과 CSV:** [조건별 평균](reports/research_validation_2026-11-11/summary/condition_means.csv), [패턴별 평균](reports/research_validation_2026-11-11/summary/pattern_means.csv), [원본 성능](reports/research_validation_2026-11-11/summary/clean_baselines.csv)
- **학습 시드 비교:** [밝은 25화소 시드 평균](reports/research_validation_2026-11-11/summary/bright25_seed_summary.csv)
- **결과·지도·학습 이력 안내:** [RESULTS_INDEX.md](reports/RESULTS_INDEX.md)

저장소에는 소스, 고정 설정, 테스트, 데이터 분할 인덱스, 결과표, 지도, 보고서와 학습·평가 요약을 포함합니다. 최종 학습 가중치는 [models/](models/README.md)에 보관합니다. 데이터셋 원본, 중간 체크포인트, 가상 환경, 전체 이미지별 예측 캐시는 포함하지 않습니다. `reports/run_summaries/`는 로컬 `runs/`의 작은 분석 산출물과 메타데이터를 복사한 스냅샷이며, `reports/training_summaries/`는 가중치를 제외한 학습 이력입니다. 스냅샷과 최종 가중치의 원본 위치 및 SHA256은 `reports/ARTIFACT_MANIFEST.json`에 기록합니다.

실행 스크립트와 기존 실행 메타데이터에는 `/home/hyochang/pixel_sensitivity` 등 당시 장비의 절대 경로가 남아 있습니다. 다른 환경에서 재실행할 때는 데이터 위치와 출력 경로를 조정해야 합니다. 기존 실행의 해시·경로 기록은 그대로 보존합니다. 아래 실행 절차는 초기 실험부터의 이력을 포함하므로 최신 분석은 위 주 결과와 [연구 검증 문서](docs/VALIDATION_RESEARCH_20261111.md)를 먼저 확인하세요.

프로포절과 `manufacturing_pixel_sensitivity_pipeline_manual.md` 30절 참조 구현을 바탕으로 작성했습니다. 원본 해상도 영상에 손상을 넣고 같은 YOLOv8n으로 clean/fault 추론, 객체 단위 평가, 민감도 지도와 GT 상자 분포를 생성합니다.

## 초기 GC10 실험 설계

- 데이터: Hugging Face `dronefreak/GC10-DET`, revision `42c086fa8dea913987febe794b1e7608b3db464d`. 비공식 Dataset Ninja 기반 재배포본입니다. 출처와 저작자 정보는 다운로드된 README 및 `data/raw/gc10_det/provenance.json`에 있습니다.
- 라벨: 같은 revision의 분할별 `metadata.jsonl`에 있는 COCO xywh를 YOLO로 변환합니다. 이미 다운로드한 개별 YOLO 라벨과도 일치 여부를 확인합니다. 원본 JPG는 수정하지 않습니다.
- 분할: filename의 마지막 frame 토큰을 제외한 접두어를 그룹으로 삼아 seed=101로 70/15/15 그룹 분할합니다. 접두어가 실제 코일/제품 ID라는 근거는 없으므로 독립 제품 분할로 단정하지 않습니다. 기본 mirror 분할은 사용하지 않습니다.
- 빈 라벨 영상은 정상 여부가 확인되지 않아 제외합니다. 정확한 decoded 픽셀 중복을 검사합니다. 근접 중복의 수동 검수는 포함하지 않습니다.
- 모델: Ultralytics 8.3.221 객체 탐지용 YOLOv8n, COCO 사전학습에서 seed=11, 최대 100 epoch, patience=20. 검증 best checkpoint 고정.
- 조건: `configs/gc10_frozen.yaml`, 4×4 grid × 11 conditions × 3 patterns. 조건은 시험 추론 전에 고정합니다. 분산 화소와 3×3 군집, 검정/255/128 치환, 부분 강도를 포함합니다.
- 평가: confidence=0.25, class-aware matching IoU=0.5. 별도로 동일 예측 캐시에서 공식 COCO mAP50 / mAP50:95를 계산합니다. 두 평가기의 대응 규칙은 다르므로 혼용하지 않습니다.
- RGB 화소의 합성 손상입니다. hot_like의 255 치환은 dark-current 물리 모델이 아닙니다. 실제 센서 RAW/ISP/DPC는 포함하지 않습니다.

## 환경

이 기기에서는 기존 CUDA PyTorch/torchvision을 재사용하는 `.venv`를 만들었습니다. 다른 기기에서는 먼저 호환되는 torch/torchvision을 설치하세요.

```bash
cd /home/hyochang/pixel_sensitivity
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-example.txt
.venv/bin/pip install huggingface_hub requests
.venv/bin/python -m pytest tests -q
```

`requirements-lock.txt`에는 이 기기에서 사용한 전체 패키지 버전이 있습니다. 기존 환경의 관련 없는 패키지도 포함하므로 타 기기에서 전체 설치할 파일은 아닙니다.

## 실행 순서

다음은 새 작업 디렉터리에서 처음 실행하는 순서입니다. 이미 만들어진 canonical 데이터나 학습 이름에 덮어쓰지 않도록 구현되어 있습니다. 조건이나 코드가 바뀌면 새 평가 출력 디렉터리를 사용하세요.

```bash
.venv/bin/python scripts/download_gc10.py
.venv/bin/python scripts/index_gc10.py
.venv/bin/python src/prepare.py \
  --index inputs/gc10_index.csv --classes inputs/gc10_classes.json \
  --dataset gc10_det --format yolo --out data/canonical/gc10_det

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  YOLO_CONFIG_DIR="$PWD/.ultralytics" .venv/bin/python src/train.py \
  --data data/canonical/gc10_det/dataset.yaml --name gc10_yolov8n_s11 \
  --weights yolov8n.pt --seed 11 --device 0 --epochs 100 --batch 16

# 실제 실행은 GPU 0/1이 손상 조건을 나누어 처리합니다.
# 두 워커가 모두 성공한 뒤 분석과 보고서 생성까지 자동 수행합니다.
.venv/bin/python scripts/evaluate_gc10.py

# 이미 추론을 완료한 뒤 분석·보고서만 재생성하려면:

.venv/bin/python src/analyze.py --run runs/evaluation/gc10_s11_frozen \
  --classes data/canonical/gc10_det/classes.json --ap --plots --bootstrap 2000
.venv/bin/python scripts/report_gc10.py
```

Inference resumes verified completed shards for exactly the same data/model/config/code/environment. Partial shards are rerun. Bootstrap intervals are pointwise and conditional on the trained checkpoint, not simultaneous intervals across the map.

## 주 결과물

- `runs/train/gc10_yolov8n_s11/weights/best.pt`: 학습 가중치
- `inputs/gc10_source_audit.json`: 배포본 점검, 제외·분할 기록
- `data/canonical/gc10_det/audit.json`: 학습/검증/시험별 이미지·클래스 수
- `runs/evaluation/gc10_s11_frozen/run_meta.json`: 실험 설정, 데이터·가중치·코드 SHA256, 패키지 버전
- `runs/evaluation/gc10_s11_frozen/predictions/`: clean 및 528개 손상 조건 shard, 실제 손상 좌표/수/값과 예측
- `runs/evaluation/gc10_s11_frozen/analysis/REPORT_KO.md`: 실험 결과 보고서
- 같은 analysis 디렉터리: `grid_summary.csv`, `condition_metrics.csv`, `condition_summary.csv`, `paired_images.csv`, `occupancy.csv`, 클래스별 지표, 민감도·분포·예시 PNG

Recall 지도에서 음수는 검출 개선을 뜻하며 그대로 남깁니다. 위치별 최대값은 사후 기술통계이고 특정 위치가 통계적으로 더 취약하다는 검정 결과는 아닙니다. 단일 데이터셋·단일 학습 시드의 실험이며 실제 카메라 수명·정비 기준·출하 불량률을 측정하지 않습니다.

## PCB-IND / NEU-DET 밝은 화소 반복 실험 (2026-11-05)

GC10과 동일한 밝은 조건 5개만 사용합니다: `bright_k1`, `bright_k9`, `bright_k25`, `bright_cluster9`, `bright_partial9`. 고정 파일은 `configs/bright_frozen.yaml`이며, 격자 4×4·패턴 반복 3회·전처리·판정 기준은 GC10과 같습니다. 각 데이터셋은 원본 `yolov8n.pt`에서 독립적으로 학습합니다.

```bash
.venv/bin/pip install gdown
.venv/bin/python scripts/download_other_datasets.py

# d=neu_det 또는 pcb_ind. 새 canonical 디렉터리에서 처음 준비할 때 실행합니다.
.venv/bin/python scripts/index_industrial.py "$d"
.venv/bin/python src/prepare.py --index "inputs/${d}_index.csv" \
  --classes "inputs/${d}_classes.json" --dataset "$d" --format yolo \
  --out "data/canonical/$d"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  YOLO_CONFIG_DIR="$PWD/.ultralytics" .venv/bin/python src/train.py \
  --data "data/canonical/$d/dataset.yaml" --name "${d}_yolov8n_s11" \
  --weights yolov8n.pt --seed 11 --device 0 --epochs 100 --batch 16 \
  > "logs/${d}_train.log" 2>&1
.venv/bin/python scripts/run_bright_experiment.py --dataset "$d" --device 0

# 두 데이터셋의 실험과 보고서가 모두 생성된 후:
.venv/bin/python scripts/report_bright_comparison.py
```

- PCB-IND: 공식 Zenodo record 19723114의 v4 archive, MD5 검증. 클래스 순서는 archive의 `classes.json`을 기준으로 합니다. COCO 좌표를 사용해 YOLO 6자리 반올림으로 인한 미세한 경계 초과를 피합니다. 빈 라벨 127장을 제외하고 공식 분할을 유지합니다. 숫자 ID는 같은데 t/b 면만 다른 자료가 분할에 걸칠 수 있어 독립 보드 시험으로 해석하지 않습니다.
- NEU-DET: 원 배포처 Google Drive archive. VOC 좌표를 1-based inclusive로 변환하고 difficult 객체를 포함합니다. 주석 충돌이 있는 동일 픽셀 영상 2장과 중복 상자 3개를 정리한 후 이미지 클래스별 70/15/15(seed=101)로 분할합니다. 실제 코일/제품 그룹은 확인되지 않았습니다.
- 원본 화소 수를 맞춘 실험입니다. NEU 200×200, PCB 300×300, GC10의 큰 원본 해상도에서 같은 25화소의 면적 비율과 resize 이후 크기는 서로 다릅니다.
- 데이터별 결과: `runs/evaluation/{neu_det,pcb_ind}_s11_bright/analysis/REPORT_KO.md`.
- 통합 결과: `reports/bright_replication_2026-11-05/COMPARISON_KO.md`.
- 진행 확인: `.venv/bin/python scripts/status_bright.py`.

이번 실행에서 NEU-DET는 `--device 1`, PCB-IND는 `--devices 0,1`로 추론했습니다. 완료 캐시를 재사용할 때는 같은 장치/worker 설정을 지정해야 합니다.

```bash
.venv/bin/python scripts/run_bright_experiment.py --dataset neu_det --device 1
.venv/bin/python scripts/run_bright_experiment.py --dataset pcb_ind --devices 0,1
```

## 밝은 화소 25개: 8×8 지도 재평가 (2026-11-10)

기존 세 모델과 전체 시험자료를 유지하고 밝은 화소 25개 조건만 8×8 격자로 실제 재추론합니다. 설정은 `configs/bright25_grid8.yaml`입니다. 데이터셋별 clean 1개 + 64위치×3패턴 = 193개 묶음을 생성합니다.

```bash
.venv/bin/python scripts/run_bright25_grid8.py
# 완료 캐시에서 지도·보고서만 다시 생성하려면:
.venv/bin/python scripts/plot_bright25_grid8.py
```

출력은 `runs/evaluation/{dataset}_s11_bright25_g8/`와 `reports/bright25_grid8_2026-11-10/`입니다. 8×8 한 칸의 면적은 기존 4×4의 약 1/4이고 격자 의존 seed로 새 패턴을 생성하므로, 같은 25개라도 더 좁은 영역에 모입니다. 기존 지도의 보간이나 손상 밀도를 일치시킨 격자 해상도 비교가 아닙니다. 16칸과 64칸의 최대값은 선택 후보 수 역시 다릅니다.

## 8×8 · 핫픽셀25개 · 분산형 무작위 배치10개

```bash
.venv/bin/python scripts/run_grid8_pattern10.py
# 모든 추론/분석이 끝난 뒤 평균표·그림만 재생성:
.venv/bin/python scripts/report_grid8_pattern10.py
# 진행 확인:
.venv/bin/python scripts/status_pattern10.py
```

고정 설정은 `configs/bright25_grid8_pattern10.yaml`입니다. 기존 P01~P03 예측은 모델·데이터·소스·환경·모든 주입/평가 규칙이 같은지 검사하고 SHA256·출처를 기록해 재사용하며, P04~P10만 추가 추론합니다. 출력은 `runs/evaluation/{dataset}_s11_bright25_g8_p10/`와 `reports/bright25_grid8_pattern10_2026-11-10/`에 저장합니다.

각 패턴은 64개 위치 평균이며 데이터셋 총평균은 10개 패턴 평균입니다. AP는 각 위치/패턴의 공식 평가값을 평균합니다. P번호는 데이터셋 내부의 난수 배치 번호이며 데이터셋 간 동일 좌표 형상을 뜻하지 않습니다. 여기서10개는 분산형 무작위 배치10개이며 군집·선형 등10가지 결함 유형을 뜻하지 않습니다.

## 연구 검증 확장 — 2026-11-11

기존 결과의 제품 결함 분포/공간 통계 분석, PCB 숫자 ID 그룹 분리, 추가 학습7개, 총27개 평가 조합을 실행합니다. 주 실험은 GC10/NEU와 새 `pcb_ind_board` 분할이며, 각자료의 시드11에서7조건을 비교하고 시드22/33에서밝은25조건을 반복합니다. 모든 조건은8×8·10분산형배치입니다.

- 계획: `docs/superpowers/plans/2026-11-11-research-validation.md`
- 재개 지침: `docs/RESEARCH_VALIDATION_RESUME.md`
- 진행: `.venv/bin/python scripts/status_research_validation.py`
- 학습 큐: `scripts/train_research_validation.py`
- 본 평가 master: `scripts/run_research_validation_compiled.py` (진행중인PID가없을때만재실행)
- 실제 평가 디렉터리: `runs/research_validation_compiled_20261111/`
- 주 manifest/state/최종COMPLETE표식: `reports/research_validation_2026-11-11/`
- 최종 결과: `reports/research_validation_2026-11-11/summary/REPORT_KO.md`

후처리 실행 최적화는 원래 FP32·batch16을 유지합니다. 원래 argsort/동점 순서를 보존하고 IoU 경계/비정상/큰 후보는 원본NMS로 되돌리는 guarded compiled NMS를 사용합니다. 기존과정의예측값동등성을검증한인증서와체크포인트별control을보존합니다. 초기eager부분실행과별도프로토타입은주27작업통계에포함하지않습니다. 연구조건, 모델,입력데이터와유효결과의SHA를기록하며, 완료상태재사용전동일성을검사합니다.
