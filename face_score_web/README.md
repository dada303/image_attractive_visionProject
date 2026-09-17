# DenseNet121 로컬 얼굴 점수 프로그램

## 실행

`start_local.cmd`를 더블클릭한 뒤 브라우저에서 http://localhost:8766 을 엽니다.
사진 선택 또는 카메라 촬영 → **점수 확인** → 실제 모델 예측 점수(1~5)를 확인합니다.
실행 창에서 Ctrl+C를 누르면 종료합니다.

새 PC에서는 Python 3.12 설치 후 `setup_local.cmd`를 한 번 실행합니다.
이 PC에는 `.venv`에 CPU용 실행 환경이 준비되어 있습니다.

## 연결된 모델

- `../DenseNet121/outputs/best_model_all.pt`
- 기존 `DenseNet121/training/model.py`의 DenseNet121 + Dropout(0.3) + Linear(1024, 1)
- RGB → 224×224 Resize → ToTensor → ImageNet 평균/표준편차 정규화
- 평가 모드와 inference_mode 사용, 모델은 시작 시 한 번 로드
- 모델 출력 원본은 API의 `raw_score`, 화면 표시 점수는 1~5 범위로 제한
- 얼굴 자동 검출/크롭은 하지 않습니다. 한 사람의 정면 얼굴 사진을 선택하세요.
- 점수는 학습 데이터 기준의 모델 예측값입니다.

## 로컬 구성

`frontend/`는 화면, `local_model.py`는 추론, `serve.py`는 화면과 API를 제공하는 로컬 프로세스입니다.
외부 배포와 DB는 사용하지 않습니다. 127.0.0.1에만 연결하며 사진은 메모리에서 처리하고 저장하지 않습니다.
`dist/`는 과거 정적 배포 결과이며 로컬 프로그램에서는 사용하지 않습니다.

- `GET /api/health`: 모델 준비 상태
- `POST /api/predict`: 이미지 바이너리 본문 → JSON 점수
- 지원: JPG/PNG/WebP, 최대 10MB 및 2,000만 화소, 한글 파일명

다른 체크포인트 또는 포트:

```powershell
.\.venv\Scripts\python.exe serve.py --port 8766 --checkpoint "C:\path\best_model_all.pt"
```

모델 파일과 `.venv`는 Git에 포함하지 않습니다. 다른 PC에서는 모델도 위 경로에 복사해야 합니다.
