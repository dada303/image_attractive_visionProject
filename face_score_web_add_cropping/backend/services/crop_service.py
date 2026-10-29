"""메모리에서 얼굴 Crop을 생성하고 확인 토큰을 발급·검증한다."""
from io import BytesIO
from pathlib import Path
import hashlib
import hmac
import secrets
import sys
from threading import Lock
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import PROJECT, MAX_BYTES, MAX_PIXELS
from ..common.errors import CropError

sys.path.insert(0, str(PROJECT))
from face_score_web_add_cropping.cropping.crop_local_idol_faces_haar import largest_face, square_box

class CropService:
    """얼굴 검출기와 서명 비밀값을 재사용하는 메모리 기반 Crop 서비스."""
    def __init__(self):
        # FileStorage MEMORY avoids OpenCV filename issues with Korean Windows paths.
        """얼굴 검출기, 검출 잠금, 토큰 서명 비밀값을 한 번 초기화한다."""
        source = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        storage = cv2.FileStorage(source.read_text(encoding='utf-8'), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            self.detector = cv2.CascadeClassifier()
            self.detector.read(storage.getFirstTopLevelNode())
        finally:
            storage.release()
        if self.detector.empty():
            raise RuntimeError('Haar 얼굴 검출기를 불러오지 못했습니다.')
        self.lock = Lock()
        self.secret = secrets.token_bytes(32)

    def detect_and_crop_face(self, data):
        """이미지를 검증하고 가장 큰 얼굴을 여백 있는 정사각형으로 자른다."""
        if not data or len(data) > MAX_BYTES:
            raise CropError('10MB 이하의 사진을 선택해주세요.', 413)
        try:
            with Image.open(BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise CropError('JPG, PNG, WebP 사진만 사용할 수 있습니다.')
                if source.width * source.height > MAX_PIXELS:
                    raise CropError('2,000만 화소 이하의 사진을 선택해주세요.')
                image = ImageOps.exif_transpose(source).convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise CropError('사진을 읽을 수 없습니다. 정상적인 이미지 파일을 선택해주세요.') from error
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY))
        with self.lock:
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))
        # Same policy as the existing batch cropper: largest area, one face only.
        face = largest_face(faces)
        if face is None:
            raise CropError('얼굴을 인식하지 못했습니다. 얼굴이 잘 보이는 다른 사진을 첨부해 주세요.', 422)
        box = square_box(face, image.width, image.height, margin=0.25)
        crop = image.crop(box)  # Native pixel resolution, no resize or JPEG recompression.
        return crop, {'face_box': face, 'crop_box': box, 'face_count': len(faces)}

    def prepare(self, data):
        """Crop을 PNG로 직렬화하고 해당 바이트에 연결된 확인 토큰을 발급한다."""
        crop, metadata = self.detect_and_crop_face(data)
        buffer = BytesIO()
        crop.save(buffer, format='PNG')
        png = buffer.getvalue()
        if len(png) > MAX_BYTES:
            raise CropError('Crop 이미지가 10MB를 초과합니다. 더 작은 사진을 선택해주세요.', 413)
        stamp = str(int(time.time()))
        digest = hmac.new(self.secret, stamp.encode() + b':' + png, hashlib.sha256).hexdigest()
        return png, stamp + '.' + digest, metadata

    def verify(self, data, token):
        # Signature binds confirmation to these exact PNG bytes. No storage or re-detection.
        """확인 토큰의 서명과 30분 유효 시간을 검사한다."""
        try:
            stamp, supplied = (token or '').split('.')
            age = time.time() - int(stamp)
            expected = hmac.new(self.secret, stamp.encode() + b':' + data, hashlib.sha256).hexdigest()
            valid = 0 <= age <= 1800 and hmac.compare_digest(supplied, expected)
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise CropError('Crop 확인 정보가 없거나 만료되었습니다. 사진을 다시 선택해주세요.', 409)
