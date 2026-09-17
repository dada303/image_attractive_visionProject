# classical_cv — 고전 머신비전 기반 얼굴 매력도 예측

딥러닝을 쓰지 않는 classical computer vision 파이프라인. SCUT-FBP5500 논문
(arXiv:1801.06345)의 hand-crafted baseline을 재현한다. 논문은 기하학적 특징과 Gabor(외형)
특징을 하나의 벡터로 합친 적이 없고 **항상 따로** 평가했다 — 이 파이프라인도 동일하게
Geometric only / Gabor only를 각각 평가한다 (Gabor 64UniSample+SVR가 논문 전체 최고 PC=0.8065).

## 준비

```bash
# combined/ 에서 실행
pip install opencv-contrib-python scikit-image scikit-learn scipy pandas openpyxl
python classical_cv/src/download_lbfmodel.py   # haar cascade + 68점 LBF 랜드마크 모델 다운로드
```

> 주의: 이 프로젝트 경로에 한글 폴더명이 포함되어 있어서, OpenCV 네이티브 로더는
> 절대경로에 비 ASCII 문자가 있으면 파일을 못 연다. 그래서 모든 스크립트는 반드시
> `combined/`를 현재 작업 디렉터리로 두고 실행해야 한다 (내부적으로 모델 경로를
> 상대경로로 참조함). 이미지 로딩은 `np.fromfile` + `cv2.imdecode`로 우회한다.

## 파이프라인

1. `src/build_manifest.py` — `labels_combined_1to5.xlsx` -> `data/manifest_aihub.csv`(606장),
   `data/manifest_all.csv`(4858장, aihub+scut+연예인) 생성
2. `src/face_align.py` — Haar Cascade 얼굴검출 + Facemark LBF 68점 랜드마크 + 눈 기준 정렬(256x256)
   - **SCUT-FBP5500 이미지만 예외**: 자체 검출 대신 데이터셋이 공식 제공하는 사람이 직접 찍은
     86점 랜드마크(`SCUT-FBP5500_v2/facial landmark/*.pts`)를 사용한다 (`src/scut_landmarks.py`).
     검출 실패가 없고 논문의 실제 랜드마크에 더 가깝다. aihub/연예인은 기존 검출 방식 그대로.
3. `src/features.py` — 특징 추출
   - 기하학적 비율 18차원 (논문의 18-D를 랜드마크 문헌 기준으로 자체 정의해 재현)
   - Gabor: 5스케일 x 8방향=40 feature map -> 64UniSample(8x8 균일 샘플) -> 2560차원
4. `src/extract_features.py` — 매니페스트 전체에 대해 특징 추출, 검출 실패 로그 저장
5. `src/train_eval.py` — SVR(RBF, 고정 하이퍼파라미터), 10-fold CV, PC/MAE/RMSE + 1~5 정확도/QWK.
   Geometric only / Gabor only를 **따로** 평가한다 (논문과 동일 — 특징을 합치지 않음).

## 두 시나리오

- **aihub-only**: pose/표정(입 벌림) 변화가 큰 어려운 서브셋. 얼굴 검출/랜드마크 실패율이
  높게 나올 것으로 예상되며, classical 파이프라인의 한계를 보여주는 스트레스 테스트.
- **all (aihub+scut+연예인)**: 전체 데이터. scut/연예인 사진 비중이 커서 평균 성능이
  aihub-only보다 높게 나올 가능성이 있음 — 소스별 residual도 같이 확인할 것.

```bash
python classical_cv/src/extract_features.py classical_cv/data/manifest_aihub.csv classical_cv/results/features_aihub.npz
python classical_cv/src/extract_features.py classical_cv/data/manifest_all.csv   classical_cv/results/features_all.npz

python classical_cv/src/train_eval.py classical_cv/results/features_aihub.npz
python classical_cv/src/train_eval.py classical_cv/results/features_all.npz
```

## 결과 (SVR, 고정 하이퍼파라미터, 10-fold CV)

논문은 기하/Gabor를 항상 따로 평가했지만, **사용자 요청으로 둘을 합친 융합(fusion)도 추가 비교**했다.
융합 시 기하(18-D)가 Gabor(2560-D)에 묻히지 않도록 Gabor만 따로 PCA(200)로 줄인 뒤 이어붙인다
(`train_eval.py`의 `build_fused_pipeline`, ColumnTransformer 사용).

| 특징 | aihub만 (n=511) | 전체 (n=4740) |
|---|---|---|
| Geometric only | PC 0.105 | PC 0.575 |
| Gabor only (PCA-200) | PC 0.231 | PC 0.600 |
| **Geometric + Gabor 융합** | **PC 0.239** | **PC 0.627** |

두 시나리오 모두 융합이 단독 특징보다 성능이 좋다 — 이목구비 비율과 텍스처가 서로
보완적인 정보를 담고 있다는 뜻으로 해석할 수 있다.

## 새 이미지로 테스트해보기

```bash
python classical_cv/src/train_final_model.py   # 전체 데이터로 배포용 모델 3개(geometric/gabor/fusion) 학습/저장 (최초 1회)
python classical_cv/src/predict.py --method fusion 내사진.jpg      # 기본값, 가장 성능 좋음
python classical_cv/src/predict.py --method gabor 내사진.jpg
python classical_cv/src/predict.py --method geometric 내사진.jpg
```

### 실행파일(.exe)로 만들기

Python이 없는 팀원도 쓸 수 있도록 단일 실행파일로 빌드할 수 있다 (PyInstaller).
결과물이 약 200MB라 **git에는 커밋하지 않는다** (GitHub 파일당 100MB 제한도 넘는다) —
빌드해서 압축 후 별도로(구글 드라이브, GitHub Release 등) 공유할 것.

```bash
pip install pyinstaller
python classical_cv/src/train_final_model.py   # 모델이 없다면 먼저 학습

python -m PyInstaller --onefile --name predict_attractiveness \
  --paths classical_cv/src \
  --add-data "classical_cv/models/haarcascade_frontalface_default.xml;models" \
  --add-data "classical_cv/models/lbfmodel.yaml;models" \
  --add-data "classical_cv/models/svr_geometric.pkl;models" \
  --add-data "classical_cv/models/svr_gabor.pkl;models" \
  --add-data "classical_cv/models/svr_fused.pkl;models" \
  --distpath classical_cv/dist --workpath classical_cv/build \
  classical_cv/src/predict.py
```

`classical_cv/dist/predict_attractiveness.exe` 생성됨. 터미널에서 실행하거나
(`predict_attractiveness.exe --method fusion 사진.jpg`), 이미지 파일을 exe 위로
드래그&드롭해도 동작한다(기본값 fusion 사용). Python 환경 없이도 동작하며, 어느
폴더에서 실행해도 무관하다.

## 참고 (논문 원문 수치, Table IV/V/VI 직접 확인)

**논문은 기하학적 특징과 Gabor 특징을 절대 하나로 합치지 않았고, 항상 따로 평가했다.**
LR/GR/SVR 세 예측모델을 각 특징에 따로 적용해 비교했다 (이 저장소는 SVR만 사용).

| 특징 | 모델 | PC (전체 데이터셋 기준) |
|---|---|---|
| Geometric (18-D) | LR/GR/SVR | 0.5948 / 0.6738 / 0.6668 |
| Gabor 86-keypoints | GR/SVR | 0.7472 / 0.6691 |
| Gabor 64UniSample | GR/SVR | 0.6764 / **0.8065** (논문 전체 classical 최고) |
| CNN(ResNeXt-50, 참고) | — | 0.8997 |
