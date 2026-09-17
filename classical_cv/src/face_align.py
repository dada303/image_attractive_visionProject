"""고전 방법 기반 얼굴 검출 + 68점 랜드마크 + 정렬.

- 검출: OpenCV Haar Cascade (Viola-Jones, 2001)
- 랜드마크: OpenCV Facemark LBF (Local Binary Features, 68점, dlib과 동일한 랜드마크 배치)
- 정렬: 양쪽 눈 중심을 기준으로 회전/스케일을 정규화하는 유사변환(similarity transform)

주의: 이 프로젝트 경로에 한글 폴더명(사용자 홈 디렉터리)이 포함되어 있는데,
OpenCV의 네이티브 파일 로더(CascadeClassifier, FileStorage, imread 등)는 절대경로에
비 ASCII 문자가 있으면 파일을 열지 못하는 버그가 있다. 그래서 모델 파일은 반드시
"combined/"를 현재 작업 디렉터리로 둔 상태의 상대경로로 로드해야 한다.
(이미지 로드는 features.py 등에서 np.fromfile + cv2.imdecode로 우회한다.)
"""
import cv2
import numpy as np

CASCADE_PATH = "classical_cv/models/haarcascade_frontalface_default.xml"
LBF_MODEL_PATH = "classical_cv/models/lbfmodel.yaml"

ALIGNED_SIZE = 256
LEFT_EYE_TARGET = (0.35, 0.35)
RIGHT_EYE_TARGET = (0.65, 0.35)

# 68점 랜드마크 인덱스(dlib 표준과 동일한 배치)
JAW = list(range(0, 17))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
NOSE = list(range(27, 36))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))


def align_from_eyes(image, left_eye, right_eye):
    """양쪽 눈 좌표만 있으면(검출 방식이든 랜드마크 제공 데이터든) 정렬할 수 있게
    FaceAligner._align 로직을 독립 함수로 분리한 것. SCUT 공식 랜드마크처럼
    별도 검출 없이 눈 위치를 이미 알고 있는 경우에 재사용한다."""
    left_eye = np.asarray(left_eye, dtype=np.float64)
    right_eye = np.asarray(right_eye, dtype=np.float64)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    dist = np.linalg.norm(right_eye - left_eye)
    target_dist = (RIGHT_EYE_TARGET[0] - LEFT_EYE_TARGET[0]) * ALIGNED_SIZE
    scale = target_dist / dist if dist > 1e-6 else 1.0

    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tx = ALIGNED_SIZE * LEFT_EYE_TARGET[0] - eyes_center[0]
    ty = ALIGNED_SIZE * LEFT_EYE_TARGET[1] - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty

    aligned = cv2.warpAffine(image, M, (ALIGNED_SIZE, ALIGNED_SIZE), flags=cv2.INTER_CUBIC)
    return aligned, M


class FaceAligner:
    def __init__(self, cascade_path=CASCADE_PATH, lbf_path=LBF_MODEL_PATH):
        """cascade_path/lbf_path를 넘기면 기본 상대경로 대신 그 경로를 쓴다.
        (PyInstaller로 묶은 실행파일처럼 cwd 관례를 쓸 수 없는 경우를 위함 —
        이 경우에도 경로 문자열 자체는 반드시 ASCII여야 한다, 모듈 docstring 참고.)"""
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise FileNotFoundError(
                f"Haar cascade를 찾을 수 없습니다: {cascade_path}\n"
                "combined/ 디렉터리에서 실행 중인지 확인하고, "
                "classical_cv/src/download_lbfmodel.py 를 먼저 실행하세요."
            )
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(lbf_path)

    def detect_and_align(self, image_bgr):
        """이미지 -> (정렬된 얼굴 이미지, 정렬된 좌표계의 68점 랜드마크) 또는 (None, None)"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None, None

        # 가장 큰 얼굴을 사용
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        faces_arr = np.array([faces[0]])

        ok, landmarks = self.facemark.fit(image_bgr, faces_arr)
        if not ok or len(landmarks) == 0:
            return None, None

        pts = np.asarray(landmarks[0]).reshape(-1, 2)  # (68, 2)

        left_eye = pts[LEFT_EYE].mean(axis=0)
        right_eye = pts[RIGHT_EYE].mean(axis=0)

        aligned, M = align_from_eyes(image_bgr, left_eye, right_eye)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        aligned_pts = (M @ pts_h.T).T

        return aligned, aligned_pts
