"""고전 특징 추출: 기하학적 비율(geometric ratio) + Gabor 텍스처 (SCUT-FBP5500 논문 재현).

논문(arXiv:1801.06345)은 정확한 18차원 비율의 정의를 공개하지 않았으므로,
이 모듈에서는 얼굴 미학 문헌에서 흔히 쓰이는 18개의 랜드마크 비율을 자체 정의해서 재현한다.
Gabor는 논문 그대로 5스케일 x 8방향 = 40개 특징맵을 만들고,
"64UniSample"과 동일한 방식(각 맵을 8x8=64 지점에서 균일 샘플링) -> 40*64=2560차원으로 재현한다.
"""
import numpy as np
import cv2
from skimage.filters import gabor_kernel
from scipy.signal import fftconvolve

from face_align import (
    JAW, RIGHT_EYEBROW, LEFT_EYEBROW, NOSE, RIGHT_EYE, LEFT_EYE, MOUTH
)

# ---------------------------------------------------------------------------
# 1) 기하학적 비율 특징 (18차원)
# ---------------------------------------------------------------------------


def _dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def geometric_features(pts):
    """68점 랜드마크(정렬된 좌표계)로부터 18개의 얼굴 비율 특징을 계산한다."""
    face_width = _dist(pts[0], pts[16])
    face_height = _dist(pts[27], pts[8])
    jaw_width = _dist(pts[4], pts[12])

    left_eye_c = pts[LEFT_EYE].mean(axis=0)
    right_eye_c = pts[RIGHT_EYE].mean(axis=0)
    eye_distance = _dist(left_eye_c, right_eye_c)
    left_eye_w = _dist(pts[42], pts[45])
    right_eye_w = _dist(pts[36], pts[39])

    nose_width = _dist(pts[31], pts[35])
    nose_height = _dist(pts[27], pts[33])

    mouth_width = _dist(pts[48], pts[54])
    mouth_height = _dist(pts[51], pts[57])

    eyebrow_l = _dist(pts[22], pts[26])
    eyebrow_r = _dist(pts[17], pts[21])

    upper_face = _dist(pts[27], pts[33])  # 미간~코끝
    lower_face = _dist(pts[33], pts[8])   # 코끝~턱끝

    eps = 1e-6
    feats = [
        face_width / (face_height + eps),
        jaw_width / (face_width + eps),
        eye_distance / (face_width + eps),
        (left_eye_w + right_eye_w) / 2 / (eye_distance + eps),
        nose_width / (face_width + eps),
        nose_height / (face_height + eps),
        mouth_width / (face_width + eps),
        mouth_height / (mouth_width + eps),
        mouth_width / (nose_width + eps),
        (eyebrow_l + eyebrow_r) / 2 / (face_width + eps),
        upper_face / (lower_face + eps),
        eye_distance / (nose_width + eps),
        mouth_width / (jaw_width + eps),
        _dist(pts[36], pts[45]) / (face_width + eps),         # 눈 바깥쪽 폭
        _dist(pts[8], pts[57]) / (face_height + eps),         # 턱~입술 아래
        _dist(pts[21], pts[22]) / (eye_distance + eps),       # 미간 폭
        _dist(pts[31], pts[48]) / (nose_width + eps),         # 코~입 좌측
        _dist(pts[35], pts[54]) / (nose_width + eps),         # 코~입 우측
    ]
    return np.array(feats, dtype=np.float64)


GEOMETRIC_DIM = 18

# ---------------------------------------------------------------------------
# 2) Gabor 텍스처 특징 (40 feature maps -> 64UniSample -> 2560차원)
# ---------------------------------------------------------------------------

GABOR_FREQS = [0.05, 0.1, 0.15, 0.2, 0.25]   # 5 scales
GABOR_THETAS = [i * np.pi / 8 for i in range(8)]  # 8 orientations
UNISAMPLE_GRID = 8  # 8x8 = 64 sample points per map


def _build_gabor_kernels():
    kernels = []
    for freq in GABOR_FREQS:
        for theta in GABOR_THETAS:
            kernels.append(gabor_kernel(freq, theta=theta))
    return kernels


_GABOR_KERNELS = _build_gabor_kernels()


def gabor_features(gray_face):
    """5스케일x8방향 Gabor 필터 -> 40개 맵 -> 각 맵에서 8x8 균일 샘플링 -> 2560차원."""
    gray = gray_face.astype(np.float64) / 255.0
    h, w = gray.shape

    ys = np.linspace(0, h - 1, UNISAMPLE_GRID).astype(int)
    xs = np.linspace(0, w - 1, UNISAMPLE_GRID).astype(int)

    feats = []
    for kernel in _GABOR_KERNELS:
        real = fftconvolve(gray, np.real(kernel), mode="same")
        mag = np.abs(real)
        sampled = mag[np.ix_(ys, xs)].reshape(-1)  # 64
        feats.append(sampled)
    return np.concatenate(feats)  # 40*64 = 2560


GABOR_DIM = len(GABOR_FREQS) * len(GABOR_THETAS) * UNISAMPLE_GRID * UNISAMPLE_GRID


# ---------------------------------------------------------------------------
# 통합
# ---------------------------------------------------------------------------


def extract_all_features(aligned_bgr, pts):
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    geo = geometric_features(pts)
    gab = gabor_features(gray)
    return geo, gab
