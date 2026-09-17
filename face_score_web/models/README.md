# 로컬 추론 모델 관리

웹 프로그램은 이 폴더의 체크포인트와 각 config.json을 사용합니다. 원본 학습 파일은 기존 위치에 유지했습니다.

```text
models/
├── densenet121/
│   ├── config.json
│   └── best_model_all.pt
├── mobilenetv3/
│   ├── config.json
│   └── mobilenetv3_regression_final.pt
└── efficientnet_b0/
    ├── config.json
    └── EfficientNetAllImage.pt
```

## 모델 교체

1. 로컬 프로그램을 종료합니다.
2. 해당 모델 폴더의 pt 파일을 새 체크포인트로 교체합니다.
3. 파일명이 달라지면 config.json의 checkpoint 값을 수정합니다.
4. 전처리나 출력 변환이 달라졌다면 config.json도 학습 설정과 맞춥니다.
5. start_local.cmd로 다시 실행합니다. 모델은 시작 시 읽어 메모리에 유지됩니다.

같은 아키텍처와 출력층의 가중치로 교체해야 합니다. 출력층 구조가 바뀌면 local_model.py도 수정해야 합니다.
가중치는 strict=True, weights_only=True로 로드합니다. 실패한 모델은 콤보박스에서 사용 불가로 표시되며 실행 창에 원인이 출력됩니다.
pt 파일은 Git에서 제외되므로 다른 PC로 옮길 때 별도로 복사합니다.

| 모델 | 아키텍처 | 전처리 | 점수 변환 | 확인 근거 |
|---|---|---|---|---|
| DenseNet121 | DenseNet121 + Dropout + Linear(1024,1) | 224×224 bilinear | 직접 회귀 | DenseNet121/training |
| MobileNetV3 | MobileNetV3 Large + Linear(1280,1) | 224×224 bilinear | 직접 회귀 | train_mobilenetv3_regression.ipynb |
| EfficientNet-B0 | EfficientNet-B0 + Linear(1280,1) | 짧은 변 256 bicubic, 중앙 224 crop | 1 + 4 × sigmoid | 체크포인트 config |

공통: EXIF 방향 보정, RGB, ImageNet 정규화, 화면 점수 1~5 범위 제한. 웹 입력은 Haar 얼굴 Crop 확인 후 추론합니다.
기존 manifest.template.json은 과거 브라우저 모델 연결용 참고 파일이며 로컬 추론에는 사용하지 않습니다.
