# Colab EfficientNet-B0 학습 환경

## 시작 방법

1. Google Drive의 `MyDrive/image_attractive_visionProject/`에 원본 `images/` 폴더와 `labels_combined_1to5.xlsx`를 업로드합니다. 폴더명·파일명은 유지합니다.
2. Google Colab에서 **파일 → 노트북 업로드**로 이 폴더의 `efficientnet_b0_colab.ipynb`를 엽니다.
3. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
4. 셀을 위에서부터 실행하고 Google Drive 접근을 연결합니다.
5. 설정 셀의 `PROJECT`, `IMAGES`, `SHEET`가 실제 Drive 위치·시트와 일치하는지 확인합니다. Windows의 C: 경로는 Colab에서 직접 접근할 수 없습니다.
6. 검증·분할 셀 결과를 확인한 다음 학습 셀을 실행합니다. 결과는 Drive의 `colab_outputs/<실행 ID>/`에 저장됩니다.

원본 이미지·엑셀은 수정하지 않습니다. Drive 업로드 및 Colab의 계정 연결은 해당 계정에서 진행해야 합니다. 이 작업에서는 업로드나 GPU 학습을 실행하지 않았습니다.

## 학습 설정

| 설정 | 값 | 설명 |
|---|---|---|
| DATA_MODE | all_domains | 네 도메인 전체 |
| DATA_MODE | aihub_only | AIHub 행만 선택 |
| TASK | regression | 기본. 소수 점수를 보존해 1~5 점수 예측 |
| TASK | classification | 정수 1~5 라벨만 허용. 5개 클래스 분류 |
| EPOCHS | 15 | 최대 학습 epoch |
| BATCH_SIZE | 32 | 메모리 부족 시 16 또는 8 |
| LR | 0.0001 | 전체 백본 미세조정 학습률 |
| PATIENCE | 4 | 검증 MAE 개선이 없을 때 조기 종료 |

이미지 분류용 EfficientNet-B0의 ImageNet 사전학습 가중치를 가져와 마지막 층을 교체합니다. 회귀는 출력 하나와 MSE, 분류는 출력 다섯 개와 cross entropy를 사용합니다. 소수 점수는 임의로 반올림하지 않습니다. 분류 모델도 확률의 기댓값으로 연속 점수를 반환합니다.

## 데이터 계약

- 필수 열: `dataset`, `filename`, `score`, `score_scale`.
- 도메인: `aihub`, `scut_fbp5500`, `남자연예인`, `여자연예인`.
- `filename`: images 기준 상대 경로 또는 유일한 파일명. 중복 파일명은 상대 경로로 지정합니다.
- `score`: 누락 없는 숫자 1~5.
- `score_scale`: 누락 여부와 값 분포를 확인합니다. 형식은 실제 엑셀을 확인하지 못했으므로 자동 변환하지 않습니다. 사용자가 값 분포를 보고 모든 행이 동일한 1~5 척도인지 확인해야 합니다.
- 선택 열 `person_id`: 같은 인물은 도메인과 관계없이 같은 ID를 사용합니다.

파일 연결 실패·손상 이미지·동일 이미지의 상충 점수·중복 파일 라벨은 학습 전에 중단합니다. 중복 검사는 EXIF 방향을 적용한 RGB 픽셀 기준이며, 크기 변경·재압축 등 유사 이미지를 모두 찾아내지는 않습니다.

전체 데이터에 공통 그룹 분할을 수행한 뒤 모드를 적용합니다. 동일 이미지 또는 동일 person_id는 같은 분할에 넣습니다. person_id가 없으면 같은 인물의 다른 사진이 섞일 수 있으므로 평가 결과는 예비 결과로 취급해야 합니다. 분할은 그룹 기준 약 70/15/15이며 도메인·점수 층화는 하지 않습니다. 표시되는 분포를 확인하세요. 두 모드 비교 시 데이터 내용·행 순서·seed·그룹 열을 유지하세요.

현재 전처리는 얼굴 중심 이미지에 대한 resize/center crop과 학습 시 수평 반전입니다. 얼굴 검출·정렬 기능은 포함하지 않습니다. 전신/다중 얼굴 사진은 학습 전에 대상 얼굴 이미지로 준비하는 것이 필요합니다.

## 저장 결과

- `config.json`, `environment.txt`: 설정과 설치 버전
- `validated_manifest.csv`: 이미지 연결·해시 검사 결과
- `validation_errors.csv`, `corrupt_images.csv`: 오류 목록
- `all_domain_splits.csv`, `train.csv`, `val.csv`, `test.csv`: 분할 목록
- `history.csv`: 학습 손실·검증 MAE
- `best.pt`: 최적 검증 MAE의 가중치·설정
- `test_predictions.csv`, `test_metrics.json`: 테스트 예측·전체/도메인별 성능

## 결과
| 학습 데이터 | Test n | MSE ↓ | MAE ↓ | RMSE ↓ | Pearson r ↑ |
|---|---:|---:|---:|---:|---:|
| **모든 데이터 활용** | 729 | 0.2937 | 0.4037 | 0.5419 | **0.7946** |
| **AI Hub만 활용** | 95 | 0.7891 | 0.7087 | 0.8883 | 0.4902 |
| **SCUT만 활용** | 594 | **0.2042** | **0.3518** | **0.4519** | **0.7946** |