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

매 실행은 새 폴더를 사용합니다. best.pt는 추론용이며 optimizer 상태를 포함한 중단 지점 학습 재개 기능은 없습니다. 세션 종료 후 이전 모델을 사용하려면 노트북 마지막 셀의 CHECKPOINT에 기존 best.pt 경로를 지정하세요. 모델 정의와 전처리 셀도 필요합니다. 직접 생성한 신뢰할 수 있는 체크포인트만 로드하세요.

GPU 런타임의 torch/torchvision 설치 조합을 유지하며, 나머지 패키지는 첫 셀에서 설치합니다. Drive의 이미지 직접 읽기가 느릴 수 있습니다. 장시간 실험에서는 여유 디스크를 확인한 후 Colab 로컬 디스크에 이미지를 복사하고 IMAGES만 바꿀 수 있습니다. 결과 OUT은 Drive에 유지하세요.

## 검증 상태와 참고

생성 환경의 Windows 실행 도구 오류 때문에 실제 엑셀 검사, Python 실행, GPU 학습 검증을 수행하지 못했습니다. 노트북 JSON 생성 구조를 검사했고, 노트북에는 데이터 검증·분할 누수 검사·첫 배치 출력/손실 검사 셀이 포함되어 있습니다. Colab 실행 시 이 검증을 통과해야 합니다.

공식 모델·전처리 문서: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html

## 파일 매칭 오류 수정

CSV 4,858행은 로컬 파일과 모두 매칭됩니다. 기존 오류 204행은 한글 파일명 전체와 일치합니다(남자연예인 64, 여자연예인 140). Drive 한글 Unicode 표현 차이가 유력하지만 Drive 파일 목록 없이는 업로드 누락 등과 구별하여 확정할 수 없습니다.

수정 노트북을 Colab에 다시 업로드하고 설정부터 실행하세요. LABELS 기본값은 labels_combined_1to5.csv입니다. CSV를 Drive 프로젝트 폴더에 올리거나 LABELS를 기존 .xlsx로 바꾸세요. 매칭 코드가 노트북 안에 포함되어 별도 Python 파일 업로드는 필요 없습니다.

매칭은 dataset 폴더 범위에서 한글 NFC 및 대소문자 정규화를 적용합니다. 실제 파일명과 점수는 변경하지 않습니다. 모호한 후보는 중단하며 다른 도메인으로 대체하지 않습니다. 실패 시 validation_errors.csv와 실제 파일 경로 및 문자 코드가 있는 image_inventory.csv를 확인하세요.

전체 로컬 4,858행 매칭, 회귀 테스트 3개, 노트북 코드 구문 검사를 통과했습니다. Colab 재실행과 GPU 학습은 미검증입니다.

## SCUT-FBP5500 단독 학습

설정 셀에서 `DATA_MODE = "scut_fbp5500_only"`로 지정하세요. 공통 분할을 유지한 채 SCUT-FBP5500 행만 train/val/test에 선택합니다. 기존 `aihub_only`와 `all_domains`도 지원합니다. 모드 변경 후 설정부터 학습·평가 셀을 순서대로 다시 실행하세요.
