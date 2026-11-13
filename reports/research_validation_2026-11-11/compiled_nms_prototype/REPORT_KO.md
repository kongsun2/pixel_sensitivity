# Compiled NMS 격리 프로토타입

운영 코드와 프로세스는 변경하지 않았다. 별도GPU0프로세스에서3GiB allocator상한으로 검사했다. torchvision이 import된 경우에만 compiled경로를 타는 고정Ultralytics분기를 이용했으며, 기준 경로는 torchvision.ops.nms를 원래TorchNMS.nms로 명시적으로 치환했다.

기존 scores.argsort내림차순 순서를 먼저 고정하고 compiled NMS에는 엄격히 감소하는 정수 rank score만 전달한 후 원래인덱스로 복원한다. 원본 confidence값은 변경하지 않는다. 후보는 N>2048·비정상/퇴화상자·비유한값 또는 임의 쌍의IoU가 기준±1e-5이면 원래TorchNMS로 돌아간다.

- 보수적 후보 전체 예측사전 완전일치: True
- near-IoU보호가 없는 진단 경로 완전일치: True
- guard분기 집계: {'compiled_rank_preserved': 370, 'near_threshold': 35, 'invalid_or_degenerate': 1}
- 입력 후보수: {'n_calls': 406, 'min': 2, 'median': 109.5, 'max': 312}
- Peak allocated/reserved: 0.331/0.449GiB

고정32검증영상의clean/hot25/dark25/cluster25와17장부분배치, 부분배치후재실행을 검사했다. 동점점수·정확한IoU경계·퇴화·음수좌표의 합성 인덱스 검사도 통과했다.

단위ms, 워밍업 후 입력조건과 경로순서를 교대한8회 중앙값:

| 경로 | 전체 batch | 전처리 | forward | 후처리+pack |
|---|---:|---:|---:|---:|
| reference_pure_torch | 826.445 | 22.218 | 6.859 | 798.605 |
| guarded_rank_compiled | 172.326 | 21.693 | 6.749 | 142.789 |
| rank_compiled_diagnostic | 40.803 | 21.481 | 6.860 | 12.533 |


현재GPU는 운영작업과 공유하므로 격리속도비나 모든 데이터/가중치에 대한 정확성 보장이 아니다. [torchvision0.24공식문서](https://docs.pytorch.org/vision/0.24/generated/torchvision.ops.nms.html)는 score동점의선택비보장과IoU기준을 설명한다. rank처리와경계fallback은 그차이를 보수적으로 다루기위한 제한된 프로토타입이며 운영에 통합하지 않았다.
