"""임의의 얼굴 이미지를 넣으면 매력도 점수(1~5)를 예측하는 CLI 테스터.

학습된 모델은 classical_cv/src/train_final_model.py 로 미리 만들어둬야 한다
(classical_cv/models/svr_geometric.pkl, svr_gabor.pkl, svr_fused.pkl).

사용법 (반드시 combined/ 에서 실행):
    python classical_cv/src/predict.py --method gabor 내사진.jpg
    python classical_cv/src/predict.py --method geometric 내사진.jpg
    python classical_cv/src/predict.py --method fusion 내사진.jpg

--method:
    gabor      Gabor 텍스처(2560-D, PCA-200) 기반 SVR. 논문 재현, 성능 준수(PC~0.6대).
    geometric  얼굴 비율 18차원 기반 SVR. 더 해석하기 쉽지만 성능은 더 낮음(PC~0.55대).
    fusion     기하+Gabor 융합. 지금까지 실험 중 가장 성능이 좋음(PC~0.63대).

실행파일(.exe)로 빌드된 경우, 리소스(cascade/랜드마크 모델/학습된 pkl)는 이 파일과
함께 번들되고, 이 프로젝트 경로의 한글 때문에 생기는 OpenCV 버그를 피하기 위해
최초 실행 시 순수 ASCII 경로(%SystemRoot%\\Temp)로 한 번 복사해서 그 경로로 로드한다.
"""
import argparse
import os
import shutil
import sys

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_align import FaceAligner  # noqa: E402
from features import geometric_features, gabor_features  # noqa: E402

IS_FROZEN = getattr(sys, "frozen", False)

RESOURCE_FILES = [
    "haarcascade_frontalface_default.xml",
    "lbfmodel.yaml",
    "svr_geometric.pkl",
    "svr_gabor.pkl",
    "svr_fused.pkl",
]


def resolve_resource_dir():
    """dev 모드: classical_cv/models (상대경로, cwd=combined/ 가정).
    frozen(exe) 모드: PyInstaller가 풀어놓은 임시폴더(_MEIPASS)는 이 사용자 환경에서
    한글 경로가 섞여 OpenCV가 못 읽으므로, %SystemRoot%\\Temp 밑 ASCII 경로로 복사한다."""
    if not IS_FROZEN:
        return "classical_cv/models"

    bundle_dir = os.path.join(sys._MEIPASS, "models")
    safe_dir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Temp", "classical_cv_bundle")
    os.makedirs(safe_dir, exist_ok=True)
    for name in RESOURCE_FILES:
        src = os.path.join(bundle_dir, name)
        dst = os.path.join(safe_dir, name)
        if not os.path.isfile(src):
            continue
        if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst)
    return safe_dir


MODEL_FILES = {
    "geometric": "svr_geometric.pkl",
    "gabor": "svr_gabor.pkl",
    "fusion": "svr_fused.pkl",
}


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def predict(image_path, method):
    resource_dir = resolve_resource_dir()
    cascade_path = os.path.join(resource_dir, "haarcascade_frontalface_default.xml")
    lbf_path = os.path.join(resource_dir, "lbfmodel.yaml")
    model_path = os.path.join(resource_dir, MODEL_FILES[method])

    if not os.path.isfile(model_path):
        print(f"[오류] 학습된 모델이 없습니다: {model_path}")
        if not IS_FROZEN:
            print("먼저 실행하세요: python classical_cv/src/train_final_model.py")
        return 1

    model = joblib.load(model_path)

    img = load_image(image_path)
    aligner = FaceAligner(cascade_path=cascade_path, lbf_path=lbf_path)
    aligned, pts = aligner.detect_and_align(img)
    if aligned is None:
        print("[실패] 얼굴을 찾지 못했습니다.")
        print("정면에 가깝고, 입을 다물고, 얼굴 전체(이마~턱)가 잘리지 않은 사진으로 다시 시도해보세요.")
        return 1

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    geo = geometric_features(pts)
    gab = gabor_features(gray)
    if method == "geometric":
        feat = geo.reshape(1, -1)
    elif method == "gabor":
        feat = gab.reshape(1, -1)
    else:
        feat = np.hstack([geo, gab]).reshape(1, -1)

    score = float(model.predict(feat)[0])
    score_clipped = max(1.0, min(5.0, score))

    print(f"이미지: {image_path}")
    print(f"방법: {method} + SVR (전체 데이터셋으로 학습)")
    print(f"예측 매력도 점수: {score_clipped:.2f} / 5")
    print("(참고: 이 파이프라인의 10-fold 교차검증 성능은 PC 기준 0.55~0.66 수준입니다.")
    print(" 절대적인 평가가 아니라, 학습 데이터의 채점 경향을 참고용으로 근사한 값입니다.)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="얼굴 이미지 매력도 예측 (classical CV)")
    parser.add_argument("image", help="예측할 이미지 파일 경로")
    parser.add_argument(
        "--method", choices=["geometric", "gabor", "fusion"], default="fusion",
        help="사용할 특징/모델 (기본값: fusion)"
    )
    args = parser.parse_args()

    try:
        rc = predict(args.image, args.method)
    except Exception as e:  # noqa: BLE001
        print(f"[오류] {e}")
        rc = 1

    if IS_FROZEN:
        input("\n엔터 키를 누르면 종료합니다...")
    sys.exit(rc)


if __name__ == "__main__":
    main()
