"""업로드 요청의 파일명·전송 형식·본문 길이를 검사한다."""
from pathlib import Path
from urllib.parse import unquote
from ..config import MAX_BYTES
from .errors import CropError

def validate_upload(filename, content_type, length):
    """업로드 확장자, 바이너리 전송 형식, 크기 제한을 검사한다."""
    if Path(unquote(filename or '')).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
        raise CropError('JPG, PNG, WebP 확장자의 사진을 선택해주세요.')
    if (content_type or '').split(';')[0] != 'application/octet-stream':
        raise CropError('이미지 바이너리 요청이 필요합니다.', 415)
    if not length:
        raise CropError('빈 이미지입니다.')
    if length < 0 or length > MAX_BYTES:
        raise CropError('10MB 이하의 사진을 선택해주세요.', 413)
