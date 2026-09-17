## 학습 설정

| 설정 | 값 | 설명                       |
|---|---|--------------------------|
| DATA_MODE | all_domains | 도메인 전체                   |
| DATA_MODE | aihub_only | AIHub 행만 선택              |
| DATA_MODE | scut_fbp5500_only | scut_fbp5500 행만 선택       |
| TASK | regression | 소수 점수를 보존해 1~5 점수 예측     |
| TASK | classification | 정수 1~5 라벨만 허용. 5개 클래스 분류 |
| EPOCHS | 15 | 최대 학습 epoch 설정           |
| BATCH_SIZE | 32 | 메모리 부족 시 16 , 8          |
| LR | 0.0001 | 전체 백본 미세조정 학습률           |
| PATIENCE | 4 | 검증 MAE 개선이 없을 때 조기 종료    |

## 데이터셋

- 필수 열: `dataset`, `filename`, `score`, `score_scale`.
- 도메인: `aihub`, `scut_fbp5500`, `남자연예인`, `여자연예인`.

## 결과 (모델 보완 전)
| 학습 데이터 | Test n | MSE ↓ | MAE ↓ | RMSE ↓ | Pearson r ↑ |
|---|---:|---:|---:|---:|---:|
| **모든 데이터 활용** | 729 | 0.2937 | 0.4037 | 0.5419 | **0.7946** |
| **AI Hub만 활용** | 95 | 0.7891 | 0.7087 | 0.8883 | 0.4902 |
| **SCUT만 활용** | 594 | **0.2042** | **0.3518** | **0.4519** | **0.7946** |