# CUDA graph 격리 실행 프로토타입

상태: complete. 운영 실행 경로는 변경하지 않았다. 기존 GC10 체크포인트와 고정 검증영상32장을 사용했다. clean→hot25→dark25→cluster25와17장 부분배치 및 부분배치 후 다시 전체배치를 비교했다.

- 전체 예측 사전 완전 일치: True
- 이미지별 검출 판정 일치: True
- 보존한 anchor 상수 불변: True
- Peak allocated/reserved: 0.539/1.113GiB
- 전체 배치 중앙 wall ms: {'eager': 784.9950920790434, 'graph': 828.5775687545538}
- forward+graph입력복사 중앙 wall ms: {'eager': 6.83857686817646, 'graph': 6.855949759483337}

동일GPU에서 운영 작업이 함께 실행되므로 시간은 격리 속도 향상의 증거가 아니다. 교대로 워밍된 반복을 기록했지만 표본 수도 작다. 입력전처리와후처리는 고정Ultralytics predictor를 사용했고 CUDA graph는 batch16 forward만 캡처했다. 부분배치는 eager로 처리했다. 모델Detect의 anchor참조, static입력과출력을 보존했고 NMS에는 캡처출력의 복사본을 넘겼다.

[PyTorch2.9공식문서](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)의 side-stream워밍업과 같은 메모리 주소·고정 shape·장기참조 규칙을 따랐다. 이 제한된 결과만으로 운영 적용 또는 모든 새 체크포인트/조건의 동등성을 보장하지 않는다.
