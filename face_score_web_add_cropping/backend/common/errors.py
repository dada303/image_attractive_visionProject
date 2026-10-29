"""Crop 실패 메시지와 기존 HTTP 상태 코드를 정의한다."""
class CropError(ValueError):
    """Crop 처리 실패와 응답 상태를 전달하는 예외."""
    def __init__(self, message, status=400):
        """오류 메시지와 기존 HTTP 상태 코드를 보관한다."""
        super().__init__(message)
        self.status = status
