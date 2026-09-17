"""Shared, memory-only Haar crop service. No ML inference occurs here."""
from io import BytesIO
from pathlib import Path
import hashlib
import hmac
import secrets
import sys
from threading import Lock
import time
from urllib.parse import unquote

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropping.crop_local_idol_faces_haar import largest_face, square_box

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

class CropError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_upload(filename, content_type, length):
    if Path(unquote(filename or '')).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
        raise CropError('JPG, PNG, WebP 확장자의 사진을 선택해주세요.')
    if (content_type or '').split(';')[0] != 'application/octet-stream':
        raise CropError('이미지 바이너리 요청이 필요합니다.', 415)
    if not length:
        raise CropError('빈 이미지입니다.')
    if length < 0 or length > MAX_BYTES:
        raise CropError('10MB 이하의 사진을 선택해주세요.', 413)


class CropService:
    def __init__(self):
        # FileStorage MEMORY avoids OpenCV filename issues with Korean Windows paths.
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
        try:
            stamp, supplied = (token or '').split('.')
            age = time.time() - int(stamp)
            expected = hmac.new(self.secret, stamp.encode() + b':' + data, hashlib.sha256).hexdigest()
            valid = 0 <= age <= 1800 and hmac.compare_digest(supplied, expected)
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise CropError('Crop 확인 정보가 없거나 만료되었습니다. 사진을 다시 선택해주세요.', 409)
