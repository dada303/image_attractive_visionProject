"""추출된 특징(.npz)으로 SVR을 학습하고 논문과 동일한 프로토콜(10-fold CV)로 평가한다.

논문은 기하학적 특징과 Gabor(외형) 특징을 하나의 벡터로 합친 적이 없고, 항상 따로 평가했다
(기하 특징 -> Table IV/V, Gabor 특징 -> Table VI). 이 스크립트는 논문 그대로의 두 가지에
더해, 사용자 요청으로 **둘을 합친 융합(fusion) 조합**도 추가로 비교한다:
  - Geometric only          (논문 Table V 재현)
  - Gabor only (PCA-200)    (논문 Table VI 재현, 64UniSample+SVR가 논문 전체 최고 PC=0.8065)
  - Geometric + Gabor 융합  (논문에 없음, 사용자 요청으로 추가 — 이목구비 비율과 텍스처를
                             함께 보는 게 더 합리적인지 확인하기 위한 비교 실험)

융합 시 기하(18-D)가 Gabor(2560-D)에 묻히지 않도록, Gabor만 따로 표준화+PCA(200차원)로
줄인 뒤에 기하 특징과 이어붙인다 (ColumnTransformer). 그냥 이어붙이고 전체에 PCA를 걸면
기하 정보가 희석되는 문제가 있어 이렇게 구성했다.

주의: 논문은 예측모델로 LR/GR/SVR 세 가지를 다 비교했지만, 이 스크립트는 SVR만 사용한다.

사용법 (combined/ 에서 실행):
    python classical_cv/src/train_eval.py classical_cv/results/features_aihub.npz
    python classical_cv/src/train_eval.py classical_cv/results/features_all.npz
"""
import sys
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score

# 주의: 프로젝트 경로에 한글이 포함돼 있어 joblib의 멀티프로세싱(loky) 백엔드가
# 리소스 트래커에서 UnicodeEncodeError를 일으킨다(n_jobs>1 사용 불가).
# 그래서 폴드마다 그리드서치를 하지 않고, 고정 하이퍼파라미터로 10-fold CV만 수행한다
# (탐색 단계이므로 우선 basic한 세팅으로 결과를 보고, 필요하면 이후에 튜닝한다).
SVR_C = 10.0
SVR_GAMMA = "scale"


def build_pipeline(use_pca, n_components=None):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("svr", SVR(kernel="rbf", C=SVR_C, gamma=SVR_GAMMA)))
    return Pipeline(steps)


def build_fused_pipeline(geo_dim, gab_dim, gabor_n_components=200):
    """기하(scale만) + Gabor(scale->PCA) 를 각자 전처리한 뒤 이어붙여서 SVR에 넣는다."""
    pre = ColumnTransformer([
        ("geo", StandardScaler(), slice(0, geo_dim)),
        ("gab", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=gabor_n_components, random_state=42)),
        ]), slice(geo_dim, geo_dim + gab_dim)),
    ])
    return Pipeline([
        ("pre", pre),
        ("svr", SVR(kernel="rbf", C=SVR_C, gamma=SVR_GAMMA)),
    ])


def evaluate(X, y, pipeline_factory, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=np.float64)

    for train_idx, test_idx in kf.split(X):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])

    pc, _ = pearsonr(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    rounded = np.clip(np.round(preds), y.min(), y.max())
    acc = float(np.mean(rounded == y))
    qwk = cohen_kappa_score(y.astype(int), rounded.astype(int), weights="quadratic")

    return {"PC": pc, "MAE": mae, "RMSE": rmse, "ACC(1~5)": acc, "QWK": qwk}, preds


def main(npz_path, out_json=None):
    data = np.load(npz_path)
    geo, gab, y = data["geometric"], data["gabor"], data["scores"]
    n = len(y)
    geo_dim, gab_dim = geo.shape[1], gab.shape[1]
    print(f"[{npz_path}] n={n}, geo={geo.shape}, gabor={gab.shape}")

    fused = np.hstack([geo, gab])
    combos = {
        "Geometric only": (geo, lambda: build_pipeline(use_pca=False)),
        "Gabor only (PCA-200)": (gab, lambda: build_pipeline(use_pca=True, n_components=200)),
        "Geometric + Gabor 융합 (사용자 요청)": (
            fused, lambda: build_fused_pipeline(geo_dim, gab_dim, gabor_n_components=200)
        ),
    }

    print(f"\n{'조합':38s} {'PC':>7s} {'MAE':>7s} {'RMSE':>7s} {'ACC':>7s} {'QWK':>7s}  time")
    print("-" * 90)
    results = {}
    for name, (X, factory) in combos.items():
        t0 = time.time()
        metrics, _ = evaluate(X, y, factory)
        dt = time.time() - t0
        results[name] = metrics
        print(
            f"{name:38s} {metrics['PC']:7.4f} {metrics['MAE']:7.4f} "
            f"{metrics['RMSE']:7.4f} {metrics['ACC(1~5)']:7.4f} {metrics['QWK']:7.4f}  {dt:5.1f}s"
        )

    if out_json:
        import json
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"n": n, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_json}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
