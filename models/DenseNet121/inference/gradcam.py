# gradcam.py
# ------------------------------------------------------------------------------------
# Grad-CAM(Gradient-weighted Class Activation Mapping)을 이용해
# "DenseNet121 모델이 얼굴의 어느 부위를 중점적으로 보고 매력도 점수를 예측했는지"를
# 시각적으로 보여주기 위한 모듈.
#
# 원리: 모델의 마지막 컨볼루션 특징맵(features)에 대해 예측 점수를 역전파(backward)하면,
# 각 채널(필터)이 최종 점수에 얼마나 영향을 주었는지 그라디언트로 알 수 있다.
# 이 그라디언트를 가중치로 각 채널의 활성화맵을 가중합하면, 이미지 중 어느 위치가
# 점수 예측에 가장 크게 기여했는지를 나타내는 히트맵(Grad-CAM)이 만들어진다.
#
# 주의: 이는 모델 내부 동작을 "근사적으로 시각화"하는 대표적인 설명 기법(XAI)일 뿐,
# 사람이 실제로 매력을 느끼는 이유와 반드시 같지는 않다. 참고용으로만 활용해야 한다.
# ------------------------------------------------------------------------------------

import numpy as np
import torch
from PIL import Image

# Grad-CAM 히트맵에서 가장 강하게 반응한 위치를 3x3 구역으로 나눠 대략적인 얼굴 부위 이름을 붙인다.
_VERTICAL_LABELS = {"top": "상단(이마·눈썹)", "mid": "중간(눈·코)", "bottom": "하단(입·턱)"}
_HORIZONTAL_LABELS = {"left": "왼쪽 ", "center": "", "right": "오른쪽 "}


def compute_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """DenseNet121의 마지막 conv 특징맵을 기준으로 Grad-CAM 히트맵과 예측 점수를 계산한다.

    Returns:
        (cam, raw_score)
        cam: 0~1로 정규화된 2차원(H,W) numpy 배열 (입력 대비 1/32 크기의 저해상도 맵).
        raw_score: 모델의 원본 예측값 (1~5 클리핑 전).
    """
    holder = {}

    def forward_hook(_module, _inp, out):
        # torchvision의 DenseNet.forward는 features(x) 직후 F.relu(..., inplace=True)를 적용하는데,
        # register_full_backward_hook을 함께 쓰면 그 in-place 연산과 충돌해 런타임 에러가 발생한다.
        # 대신 텐서에 retain_grad()만 걸어두면(비-leaf 텐서는 기본적으로 grad를 보관하지 않음),
        # in-place 연산과 충돌 없이 backward() 이후 이 텐서의 grad를 그대로 읽을 수 있다.
        out.retain_grad()
        holder["tensor"] = out

    # DenseNet121의 features 는 마지막 conv/BatchNorm까지 포함한 특징 추출부(nn.Sequential)이다.
    target_layer = model.features
    h_fwd = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        x = image_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)  # 입력까지 그라디언트가 흐르도록 함 (backbone이 동결되어 있어도 동작)

        output = model(x)
        score = output.squeeze()

        model.zero_grad(set_to_none=True)
        score.backward()

        feature_tensor = holder["tensor"]
        act = feature_tensor.detach()[0]   # (C, H, W)
        grad = feature_tensor.grad[0]      # (C, H, W)
    finally:
        # 다음 호출에서 중복으로 쌓이지 않도록 훅을 반드시 해제한다.
        h_fwd.remove()

    weights = grad.mean(dim=(1, 2))  # 채널별 중요도 = 그라디언트의 공간 평균
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.cpu().numpy(), float(score.detach().cpu().item())


def build_overlay_image(face_image: Image.Image, cam: np.ndarray, img_size: int = 224) -> Image.Image:
    """얼굴 이미지 위에 Grad-CAM 히트맵을 반투명하게 겹쳐서 보여주는 이미지를 만든다."""
    import matplotlib.cm as cm  # matplotlib의 컬러맵만 필요하므로 pyplot 없이 가볍게 사용

    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize((img_size, img_size), Image.BILINEAR)
    cam_norm = np.array(cam_resized).astype(np.float32) / 255.0

    face_resized = np.array(face_image.resize((img_size, img_size)).convert("RGB")).astype(np.float32)
    heatmap = (cm.jet(cam_norm)[:, :, :3] * 255).astype(np.float32)

    overlay = (0.55 * face_resized + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def describe_focus_region(cam: np.ndarray) -> tuple[str, str]:
    """Grad-CAM에서 가장 강하게 반응한 위치를 사람이 읽기 좋은 문구로 변환한다.

    Returns:
        (짧은 라벨: 표에 넣기 좋은 짧은 문구, 긴 설명: 전체 문장)
    """
    h, w = cam.shape
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)

    v_key = "top" if y < h / 3 else ("bottom" if y > 2 * h / 3 else "mid")
    h_key = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "center")

    short_label = f"{_HORIZONTAL_LABELS[h_key]}{_VERTICAL_LABELS[v_key]}"
    sentence = (
        f"모델은 얼굴의 {short_label} 부위를 가장 비중 있게 보고 이 점수를 예측한 것으로 보입니다. "
        f"(Grad-CAM 기반 추정이며, 실제 사람이 느끼는 판단 근거와 다를 수 있습니다.)"
    )
    return short_label, sentence
