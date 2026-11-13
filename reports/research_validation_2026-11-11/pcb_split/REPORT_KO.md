# PCB 숫자 ID 단위 재분할

기존 공식 분할은 `5810_t`와 `5810_b`를 다른 면 그룹으로 처리한다. 숫자 접두어만 묶었을 때 683개 중 124개가 여러 분할에 걸치고, 63개가 학습과 시험에 함께 나타났다.

숫자 접두어를 묶어 seed=101, 그룹 수 기준70/15/15로 분리했다. 정렬한 그룹에 NumPy default_rng 순열을 적용하며 클래스별 사후 조정이나 모델 성능 기반 선택은 하지 않았다. 첫 분할에서8개 클래스가 모든 분할에 존재했다.

| 분할 | 기존 영상 / GT | 새 영상 / GT | 새 숫자 그룹 수 |
|---|---:|---:|---:|
| train | 3718 / 4746 | 3295 / 4035 | 478 |
| val | 475 / 638 | 728 / 948 | 102 |
| test | 469 / 547 | 639 / 948 | 103 |


| 클래스 | 새 train | 새 val | 새 test |
|---|---:|---:|---:|
| mouse_bite | 566 | 141 | 137 |
| missing_copper | 1025 | 286 | 403 |
| scratch | 359 | 26 | 78 |
| spurious_copper | 485 | 118 | 57 |
| copper_burr | 880 | 162 | 132 |
| stain | 386 | 124 | 74 |
| short | 227 | 78 | 55 |
| open | 107 | 13 | 12 |


새 분할의 숫자 ID 교집합은 train–val, train–test, val–test 모두0이다. 기존 양성4662장 전체를 보존했다. 원본과 canonical PNG의 해시 및 GT 사전이 이전 canonical 자료와 모두 일치한다. 기존127개 미확인 빈 라벨 제외 정책도 유지했다. 기존 prepare.py가 decoded RGB중복 검사를 통과했다.

**이 숫자 ID는 실제 물리 보드 ID임이 확인된 값이 아니라 보수적인 그룹 대리값이다.** 새 시험 코호트의 영상 수·클래스 분포가 달라지므로 기존/신규 모델의 성능 차이를 누수 제거만의 인과적 효과라고 주장할 수 없다. 중앙 ROI 특성과 양성 자료만 사용한 기존 범위도 유지된다.

- 학습 설정 파일: `/home/hyochang/pixel_sensitivity/data/canonical/pcb_ind_board/dataset.yaml`
- [분할·클래스·GT 크기/중앙 집중 비교](split_distribution_comparison.csv)
- [영상별 이전/신규 분할](image_split_membership.csv)
- [분할 전이 수](split_transition_counts.csv)
- [검증 JSON](split_audit.json)
