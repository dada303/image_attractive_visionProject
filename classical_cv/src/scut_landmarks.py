"""SCUT-FBP5500 공식 랜드마크(.pts, 86점) 로더 + 86점 스킴 기반 기하학적 비율 특징.

SCUT-FBP5500_v2/README.txt: "We artificially labeled 86 coordinates for each face."
(사람이 직접 찍은 ground truth. 논문 Table VI의 "86-keypoints" 방식이 바로 이 랜드마크다.)

.pts 파일은 바이너리: int32(포인트 개수=86) + float32 (x,y) * 86.

86점 구조는 공식 문서가 없어 좌표를 직접 시각화해서 역추적했다(index -> 좌표 범위 확인):
  0~21   얼굴 윤곽선(정수리~턱~반대쪽 정수리, 22점). 11=턱끝, 5=왼쪽 뺨(가장 왼쪽), 17=오른쪽 뺨(가장 오른쪽)
  22~31  왼쪽 눈썹(10점)
  32~41  오른쪽 눈썹(10점)
  42~49  왼쪽 눈(8점), 42=안쪽 눈꼬리, 46=바깥쪽 눈꼬리
  50~57  오른쪽 눈(8점), 50=안쪽 눈꼬리, 54=바깥쪽 눈꼬리
  58     왼쪽 눈동자(추정)
  59     오른쪽 눈동자(추정)
  60     콧대 시작점(눈 사이, nose root)
  61~72  코(12점), 64/68=콧볼 바깥쪽, 66=코끝
  73~85  입(13점), 73=오른쪽 입꼬리, 79=왼쪽 입꼬리
"""
import os
import struct
import numpy as np

OUTLINE = list(range(0, 22))
EYEBROW_L = list(range(22, 32))
EYEBROW_R = list(range(32, 42))
EYE_L = list(range(42, 50))
EYE_R = list(range(50, 58))
PUPIL_L, PUPIL_R, NOSE_ROOT = 58, 59, 60
NOSE = list(range(61, 73))
MOUTH = list(range(73, 86))

CHIN = 11
CHEEK_L, CHEEK_R = 5, 17
JAW_L, JAW_R = 8, 14
EYE_L_INNER, EYE_L_OUTER = 42, 46
EYE_R_INNER, EYE_R_OUTER = 50, 54
NOSE_ALA_L, NOSE_ALA_R, NOSE_TIP = 64, 68, 66
EYEBROW_L_INNER, EYEBROW_R_INNER = 31, 37
MOUTH_R_CORNER, MOUTH_L_CORNER = 73, 79
MOUTH_TOP, MOUTH_BOTTOM = 76, 82


def load_pts86(path):
    """SCUT .pts 파일 -> (86, 2) ndarray. 원본 이미지 픽셀 좌표계."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<i", data[:4])[0]
    if n != 86:
        raise ValueError(f"예상치 못한 랜드마크 점 개수: {n} ({path})")
    vals = struct.unpack("<" + "f" * n * 2, data[4:4 + n * 2 * 4])
    return np.array(list(zip(vals[0::2], vals[1::2])), dtype=np.float64)


def pts86_path_for(images_dir_name_ignored, filename, landmark_dir):
    stem = os.path.splitext(filename)[0]
    return os.path.join(landmark_dir, stem + ".pts")


def _dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def geometric_features_86(pts):
    """86점 공식 랜드마크로부터 face_align.geometric_features와 동일한 정의의
    18개 비율 특징을 계산한다 (68점 버전과 동일한 순서/의미, 다른 인덱스 사용)."""
    face_width = _dist(pts[CHEEK_L], pts[CHEEK_R])
    face_height = _dist(pts[NOSE_ROOT], pts[CHIN])
    jaw_width = _dist(pts[JAW_L], pts[JAW_R])

    left_eye_c = pts[EYE_L].mean(axis=0)
    right_eye_c = pts[EYE_R].mean(axis=0)
    eye_distance = _dist(left_eye_c, right_eye_c)
    left_eye_w = _dist(pts[EYE_L_INNER], pts[EYE_L_OUTER])
    right_eye_w = _dist(pts[EYE_R_INNER], pts[EYE_R_OUTER])

    nose_width = _dist(pts[NOSE_ALA_L], pts[NOSE_ALA_R])
    nose_height = _dist(pts[NOSE_ROOT], pts[NOSE_TIP])

    mouth_width = _dist(pts[MOUTH_R_CORNER], pts[MOUTH_L_CORNER])
    mouth_height = _dist(pts[MOUTH_TOP], pts[MOUTH_BOTTOM])

    eyebrow_l = _dist(pts[EYEBROW_L[0]], pts[EYEBROW_L[4]])
    eyebrow_r = _dist(pts[EYEBROW_R[0]], pts[EYEBROW_R[4]])

    upper_face = nose_height
    lower_face = _dist(pts[NOSE_TIP], pts[CHIN])

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
        _dist(pts[EYE_L_OUTER], pts[EYE_R_OUTER]) / (face_width + eps),
        _dist(pts[CHIN], pts[MOUTH_BOTTOM]) / (face_height + eps),
        _dist(pts[EYEBROW_L_INNER], pts[EYEBROW_R_INNER]) / (eye_distance + eps),
        _dist(pts[NOSE_ALA_L], pts[MOUTH_L_CORNER]) / (nose_width + eps),
        _dist(pts[NOSE_ALA_R], pts[MOUTH_R_CORNER]) / (nose_width + eps),
    ]
    return np.array(feats, dtype=np.float64)


def eye_centers_86(pts):
    return pts[EYE_L].mean(axis=0), pts[EYE_R].mean(axis=0)
