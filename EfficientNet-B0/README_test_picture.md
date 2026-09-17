# TestPicture 예측 결과 PNG 생성

`predict_test_picture.py`는 이미 학습된 `EfficientNetAllImage.pt`로 추론합니다. 재학습하거나 모델 가중치를 변경하지 않습니다.

입력 (현재 작업 디렉터리와 무관하게 프로젝트 기준으로 찾습니다):
- `../TestPicture/1` ~ `../TestPicture/5`: 평가할 사진
- `../test_labels_1to5.xlsx`: 첫 시트의 `filename`, `score` 열
- `EfficientNetAllImage.pt`: 튜닝 노트북이 저장한 state_dict/config 체크포인트

실제 점수는 폴더 번호가 아니라 엑셀의 score를 사용합니다. 한글 파일명은 NFC 정규화 후 매칭합니다. 누락/중복/잘못된 점수는 예측 전에 오류로 중단합니다.
같은 파일명이 여러 폴더에 있으면 엑셀 filename에 `1/photo.jpg`처럼 폴더명을 포함하세요.

## 실행 (PowerShell, 프로젝트 루트)

```powershell
cd C:\sungwon\image_attractive_visionProject
.\face_score_web\.venv\Scripts\python.exe -m pip install -r .\EfficientNet-B0\requirements-predict.txt
.\face_score_web\.venv\Scripts\python.exe .\EfficientNet-B0\predict_test_picture.py
```

현재 PC는 설치와 실제 실행을 완료했습니다.

## 출력

`EfficientNet-B0/outputs/test_picture/`:
- `score_1_01.png` ~ `score_5_01.png`: 폴더별 4열 이미지 모음, 사진 아래 예측/실제 점수와 절대 오차 표시
- `predictions.csv`: 파일별 원래 정밀도의 실제/예측 점수
- `metrics.json`: MSE/MAE/RMSE/Pearson r, 체크포인트 SHA256 및 학습 설정

한 PNG에는 최대 12장, 초과하면 다음 페이지를 생성합니다. 이미지는 비율을 유지해 표시하며 모델 입력은 학습 때와 동일한 Resize 256 bicubic → CenterCrop 224 → ImageNet 정규화입니다. 출력은 `1 + 4 * sigmoid`입니다. PNG의 점수는 소수 둘째 자리까지 표시합니다.
같은 출력 경로로 다시 실행하면 같은 이름의 파일을 덮어씁니다. 입력 수를 변경해 이전 페이지가 남는 것을 피하려면 새 --output 경로를 지정하세요.

```powershell
.\face_score_web\.venv\Scripts\python.exe .\EfficientNet-B0\predict_test_picture.py --columns 4 --batch-size 8 --output .\EfficientNet-B0\outputs\new_run
```

`--checkpoint`, `--images`, `--labels`, `--device cpu`, `--font`으로 변경 가능합니다.
Windows에서는 맑은 고딕을 사용합니다. 다른 운영체제에서는 한글 TTF 파일을 --font로 지정하세요.
현재 체크포인트와 전처리/출력 계약이 다른 모델을 넣으면 설정을 확인하도록 중단합니다.

검증: 현재 50장 모두 엑셀과 일대일 매칭, PNG 5개 생성. n=50, MSE=0.9208, MAE=0.8114, RMSE=0.9596, Pearson r=0.8881. 이 값은 이번 TestPicture 평가 결과이며 학습 당시 테스트셋 지표와 구분해야 합니다.
