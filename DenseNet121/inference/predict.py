# predict.py
# ------------------------------------------------------------------------------------
# 새로운 사진 한 장을 입력받아:
#   1) OpenCV Haar Cascade로 "얼굴을 인식"하여 얼굴 영역만 잘라내고
#   2) train_aihub.py(안 1) 또는 train_all.py(안 2)로 학습해 둔 DenseNet121 모델로
#      매력도 점수(1~5)를 예측해 보여준다.
#
# 실행 예:
#   # 전체 데이터셋으로 학습한 모델(안 2)로 예측
#   python predict.py --image ./someone.jpg --model all
#
#   # AIHub 데이터셋만으로 학습한 모델(안 1)로 예측
#   python predict.py --image ./someone.jpg --model aihub
#
#   # 두 모델의 예측 결과를 한 번에 비교
#   python predict.py --image ./someone.jpg --model both
# ------------------------------------------------------------------------------------

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..training.dataset import build_transforms
from ..training.model import build_model
from .gradcam import compute_gradcam, build_overlay_image, describe_focus_region
from ..training.train_common import DEFAULT_OUTPUT_ROOT, DEFAULT_LABELS_PATH


def _load_face_cascade() -> cv2.CascadeClassifier:
    """Haar Cascade 얼굴 검출기 XML 파일을 불러온다.

    [주의] Windows에서 사용자 계정명이 한글 등 비-ASCII 문자를 포함하는 경우
    (예: C:\\Users\\한국전파진흥협회\\...), OpenCV의 내부 파일 열기 함수가 해당
    절대경로를 제대로 읽지 못해 CascadeClassifier가 빈 상태(empty)로 로드되는
    문제가 있다. 이를 우회하기 위해 1) 절대경로 2) 현재 작업 폴더 기준 상대경로
    (상대경로 문자열 자체에는 한글이 포함되지 않으므로 우회 가능) 3) 파일을
    작업 폴더에 통째로 복사한 뒤 파일명만으로 불러오기, 순서로 시도한다.
    """
    abs_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # 상대경로 문자열에는 한글 등 비-ASCII 문자가 섞이지 않는 경우가 많으므로 먼저 시도하고,
    # (드라이브가 달라 상대경로를 만들 수 없다면) 절대경로를 그다음으로 시도한다.
    candidates = []
    try:
        candidates.append(os.path.relpath(abs_path, os.getcwd()))
    except ValueError:
        pass  # 드라이브가 서로 달라 상대경로를 만들 수 없는 경우
    candidates.append(abs_path)

    for path in candidates:
        clf = cv2.CascadeClassifier(path)
        if not clf.empty():
            return clf

    # 최후 수단: 표준 파이썬 open()은 유니코드 경로를 문제없이 다루므로,
    # 파일 내용을 읽어 현재 폴더에 "파일명만" 사용해 복사한 뒤 다시 시도한다.
    with open(abs_path, "rb") as f:
        data = f.read()
    fallback_name = "_haarcascade_frontalface_default.xml"
    with open(fallback_name, "wb") as f:
        f.write(data)
    clf = cv2.CascadeClassifier(fallback_name)
    if clf.empty():
        raise RuntimeError("얼굴 검출기(Haar Cascade)를 불러오지 못했습니다.")
    return clf


def detect_and_crop_face(image_path: str, margin: float = 0.3):
    """OpenCV의 Haar Cascade 얼굴 검출기로 사진 속 얼굴을 찾아 잘라낸다.

    Args:
        image_path: 입력 이미지 경로
        margin: 검출된 얼굴 박스 주변에 추가로 포함할 여백 비율.
                DenseNet121 학습 데이터가 얼굴만이 아니라 머리카락/목 주변까지
                포함하고 있으므로, 여백을 조금 주어야 학습 데이터와 프레이밍이 비슷해진다.

    Returns:
        (PIL.Image로 잘라낸 얼굴 이미지, 얼굴 검출 여부)
        얼굴을 찾지 못하면 원본 이미지 전체를 그대로 반환한다.
    """
    # 주의: cv2.imread()는 위와 동일한 이유로 한글이 섞인 절대경로에서 실패할 수 있어,
    # 유니코드 경로를 안전하게 처리하는 PIL로 이미지를 읽은 뒤 배열로 변환한다.
    image_rgb = np.array(Image.open(image_path).convert("RGB"))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = _load_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        # 얼굴 검출 실패 여부는 호출부(predict_one)가 face_found 값으로 판단해 메시지를 출력한다.
        return Image.fromarray(image_rgb), False

    # 얼굴이 여러 명 검출되면, 가장 크게(가까이) 찍힌 얼굴 하나를 대표로 사용한다.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    h_img, w_img = image_rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w_img, x + w + mx), min(h_img, y + h + my)

    cropped = image_rgb[y1:y2, x1:x2]
    return Image.fromarray(cropped), True


def load_trained_model(run_name: str, device: torch.device, output_root: str) -> torch.nn.Module:
    """train_aihub.py / train_all.py 가 저장해 둔 체크포인트를 불러온다."""
    ckpt_path = os.path.join(output_root, run_name, f"best_model_{run_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"먼저 train_{run_name}.py 를 실행해 모델을 학습시켜야 합니다."
        )
    # 체크포인트로 가중치를 그대로 덮어쓸 것이므로 ImageNet 가중치를 새로 받을 필요는 없다.
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# run_name -> 사람이 읽기 좋은 모델 설명. predict.py CLI와 app_gui.py(GUI)가 함께 사용한다.
MODEL_LABELS = {
    "aihub": "AIHub 데이터만 학습한 모델(안 1)",
    "all": "전체 데이터셋으로 학습한 모델(안 2)",
}

# 모델은 한 번 불러오면 그대로 메모리에 재사용한다 (특히 GUI에서 버튼을 여러 번 눌러도
# 매번 체크포인트를 디스크에서 다시 읽지 않도록 하기 위함).
_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def get_cached_model(run_name: str, device: torch.device, output_root: str) -> torch.nn.Module:
    key = f"{run_name}:{output_root}:{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_trained_model(run_name, device, output_root)
    return _MODEL_CACHE[key]


def load_reference_metrics(run_name: str, output_root: str) -> dict | None:
    """해당 모델이 테스트셋에서 기록한 성능 지표(summary)를 읽어온다. 없으면 None."""
    metrics_path = os.path.join(output_root, run_name, f"test_metrics_{run_name}.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


# labels_combined_1to5.xlsx는 매 조회마다 다시 읽지 않도록 최초 1회만 불러와 재사용한다.
_LABELS_DF_CACHE: pd.DataFrame | None = None


def lookup_ground_truth(image_path: str) -> tuple[str | None, str | None, float | None]:
    """이미지가 images/<dataset>/<filename> 형태의 우리 학습 데이터셋 안에 있는 사진이라면,
    labels_combined_1to5.xlsx에서 실제(정답) 매력도 점수를 찾아 함께 반환한다.

    사용자가 새로 촬영/업로드한 사진처럼 데이터셋에 없는 사진이면 (None, 파일명, None)을 반환한다.
    (여러 장을 한 번에 측정해 HTML로 저장할 때, 데이터셋 사진은 '예측 vs 실제'를 비교해서
     모델 성능을 직관적으로 확인할 수 있게 해준다.)
    """
    global _LABELS_DF_CACHE
    if _LABELS_DF_CACHE is None:
        _LABELS_DF_CACHE = pd.read_excel(DEFAULT_LABELS_PATH)

    abs_path = os.path.abspath(image_path)
    dataset_guess = os.path.basename(os.path.dirname(abs_path))
    filename = os.path.basename(abs_path)

    match = _LABELS_DF_CACHE[
        (_LABELS_DF_CACHE["dataset"] == dataset_guess) & (_LABELS_DF_CACHE["filename"] == filename)
    ]
    if len(match) == 0:
        return None, filename, None
    return dataset_guess, filename, float(match.iloc[0]["score"])


def predict_one(
    image_path: str,
    run_name: str,
    device: torch.device,
    output_root: str,
    img_size: int = 224,
    face_detect: bool = True,
    with_gradcam: bool = True,
) -> dict:
    """이미지 한 장에 대해 지정한 모델(run_name)로 매력도 점수를 예측한다.

    CLI(predict.py)와 GUI(app_gui.py)가 동일한 예측 로직을 공유하기 위한 핵심 함수.
    점수뿐 아니라 Grad-CAM을 이용해 "모델이 얼굴의 어느 부위를 중점적으로 보고
    이 점수를 예측했는지"에 대한 히트맵 이미지와 설명 문구도 함께 만들어 반환한다.

    Args:
        with_gradcam: False로 주면 역전파 없이 순전파만으로 점수만 빠르게 계산한다.
            여러 장을 한꺼번에 측정할 때, 화면에 보여줄 대표 사진 1장만 Grad-CAM까지
            계산하고 나머지는 점수만 빠르게 계산해 전체 처리 시간을 줄이기 위해 사용한다.

    Returns:
        {
          "score": 예측 점수(float, 1~5로 클리핑됨),
          "rounded": 반올림한 정수 등급,
          "face_found": 얼굴 검출 성공 여부,
          "reference_metrics": 해당 모델의 테스트셋 성능 요약(dict) 또는 None,
          "face_image": 실제로 모델에 입력된 얼굴 이미지 (PIL.Image, 크롭된 정사각형 원본),
          "gradcam_image": 얼굴 이미지 위에 Grad-CAM 히트맵을 겹친 이미지 (PIL.Image) 또는 None,
          "reason_short": 표에 넣기 좋은 짧은 판단 부위 문구 (예: '중간(눈·코)') 또는 None,
          "reason_text": 판단 근거를 풀어 쓴 전체 문장 또는 None,
        }
    """
    if face_detect:
        face_image, face_found = detect_and_crop_face(image_path)
    else:
        face_image, face_found = Image.open(image_path).convert("RGB"), None

    transform = build_transforms(img_size=img_size, train=False)
    image_tensor = transform(face_image)

    model = get_cached_model(run_name, device, output_root)

    if with_gradcam:
        # 점수 예측과 Grad-CAM 계산을 한 번의 순전파/역전파로 함께 처리한다.
        cam, raw_score = compute_gradcam(model, image_tensor, device)
        gradcam_image = build_overlay_image(face_image, cam, img_size=img_size)
        reason_short, reason_text = describe_focus_region(cam)
    else:
        model.eval()
        with torch.no_grad():
            raw_score = model(image_tensor.unsqueeze(0).to(device)).squeeze().item()
        gradcam_image, reason_short, reason_text = None, None, None

    score = float(min(5.0, max(1.0, raw_score)))

    return {
        "score": score,
        "rounded": round(score),
        "face_found": face_found,
        "reference_metrics": load_reference_metrics(run_name, output_root),
        "face_image": face_image,
        "gradcam_image": gradcam_image,
        "reason_short": reason_short,
        "reason_text": reason_text,
    }


def format_result_text(run_name: str, result: dict) -> str:
    """predict_one() 결과를 사람이 읽기 좋은 여러 줄 문자열로 만든다 (CLI/GUI 공용)."""
    lines = [f"[{MODEL_LABELS[run_name]}]"]
    lines.append(f"  예측 매력도 점수: {result['score']:.2f} / 5.00  (반올림 등급: {result['rounded']}점)")
    ref = result["reference_metrics"]
    if ref is not None:
        lines.append(f"  (참고) '{run_name}' 모델의 테스트셋 성능: "
                      f"MAE={ref['mae']:.3f}, 정확도(±1점)={ref['accuracy_within1']*100:.1f}%")
    if result["reason_text"] is not None:
        lines.append(f"  판단 근거: {result['reason_text']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="새 이미지의 얼굴 매력도를 예측한다 (1~5점).")
    parser.add_argument("--image", required=True, help="예측할 이미지 파일 경로")
    parser.add_argument("--model", choices=["aihub", "all", "both"], default="both",
                         help="사용할 모델: 안1(aihub만 학습), 안2(전체 학습), 또는 둘 다 비교(both)")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                         help="train_*.py 가 결과를 저장한 outputs 폴더 경로 (기본값: 프로젝트 루트의 outputs/)")
    parser.add_argument("--no-face-detect", action="store_true",
                         help="얼굴 검출을 건너뛰고 이미지 전체를 그대로 모델에 입력")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    # GPU(CUDA)가 있으면 자동으로 사용하고, 없으면 CPU를 사용한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_names = ["aihub", "all"] if args.model == "both" else [args.model]

    print(f"\n입력 이미지: {args.image}")
    print("=" * 50)
    for run_name in run_names:
        result = predict_one(args.image, run_name, device, args.output_root,
                              img_size=args.img_size, face_detect=not args.no_face_detect)
        if result["face_found"] is True:
            print("[정보] 얼굴을 인식하여 해당 영역으로 잘라냈습니다.")
        elif result["face_found"] is False:
            print("[경고] 얼굴을 찾지 못했습니다. 이미지 전체를 사용해 예측합니다.")
        print(format_result_text(run_name, result))
        print("-" * 50)


if __name__ == "__main__":
    main()
