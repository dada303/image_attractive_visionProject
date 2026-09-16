"""배포용 모델을 학습해서 저장한다 (predict.py 에서 사용).

Geometric only, Gabor only, 그리고 사용자 요청으로 추가한 Geometric+Gabor 융합
세 모델을 각각 저장한다. 평가(train_eval.py)는 10-fold 교차검증으로 성능을
"측정"하는 게 목적이고, 이 스크립트는 그 성능이 나온 것과 동일한 설정으로
전체 데이터를 다 써서 실제로 새 이미지에 쓸 모델을 "학습/저장"하는 게 목적이다.

학습 데이터는 항상 전체(all) 데이터셋(aihub+scut+연예인)을 사용한다 —
aihub만 쓴 모델은 성능이 크게 낮아(PC~0.1~0.2) 실사용에 부적합하기 때문이다.

사용법 (combined/ 에서 실행):
    python classical_cv/src/train_final_model.py
"""
import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eval import build_pipeline, build_fused_pipeline  # noqa: E402

FEATURES_NPZ = "classical_cv/results/features_all.npz"
MODEL_GEOMETRIC = "classical_cv/models/svr_geometric.pkl"
MODEL_GABOR = "classical_cv/models/svr_gabor.pkl"
MODEL_FUSED = "classical_cv/models/svr_fused.pkl"


def main():
    data = np.load(FEATURES_NPZ)
    geo, gab, y = data["geometric"], data["gabor"], data["scores"]
    geo_dim, gab_dim = geo.shape[1], gab.shape[1]
    print(f"학습 데이터: n={len(y)}, geo={geo.shape}, gabor={gab.shape}")

    geo_pipe = build_pipeline(use_pca=False)
    geo_pipe.fit(geo, y)
    joblib.dump(geo_pipe, MODEL_GEOMETRIC)
    print(f"저장: {MODEL_GEOMETRIC}")

    gab_pipe = build_pipeline(use_pca=True, n_components=200)
    gab_pipe.fit(gab, y)
    joblib.dump(gab_pipe, MODEL_GABOR)
    print(f"저장: {MODEL_GABOR}")

    fused_pipe = build_fused_pipeline(geo_dim, gab_dim, gabor_n_components=200)
    fused_pipe.fit(np.hstack([geo, gab]), y)
    joblib.dump(fused_pipe, MODEL_FUSED)
    print(f"저장: {MODEL_FUSED}")


if __name__ == "__main__":
    main()
