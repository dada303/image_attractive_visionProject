# API 계약 (설계; 아직 구현되지 않음)

POST /api/v1/predict
Content-Type: multipart/form-data
file: 필수 이미지 1개
source: upload | camera (기본 upload)
모델 경로/임의 모델은 클라이언트가 지정하지 않습니다.

성공 200 예시(예시 점수이며 실제 예측 결과가 아님):
{
  "request_id": "server-generated-id",
  "score": 3.72,
  "score_scale": [1, 5],
  "model": "densenet121",
  "model_version": "v1",
  "elapsed_ms": 125
}

표시에는 소수점 두 자리, API 내부 점수는 정밀도를 유지합니다.
score 범위는 학습 출력 계약으로 검증합니다. 기존 모델 출력 규칙을 모른 채 sigmoid나 clipping을 추가하지 않습니다.

실패 응답:
{"error": {"code": "INVALID_IMAGE", "message": "이미지를 읽을 수 없습니다."}, "request_id": "..."}
400 INVALID_IMAGE: 빈 파일·손상 이미지·디코딩 실패
413 IMAGE_TOO_LARGE: 업로드 바이트 또는 픽셀 수 제한 초과
415 UNSUPPORTED_FORMAT: 지원하지 않는 실제 이미지 형식
422 INVALID_REQUEST: 요청 필드 오류
503 MODEL_NOT_READY / BUSY: 모델 미준비 또는 처리량 초과
500 INFERENCE_FAILED: 모델 추론 실패; 서버 경로/stack trace는 응답하지 않음

GET /api/v1/health/live → 프로세스 생존
GET /api/v1/health/ready → 모델 로드 및 추론 준비 여부
분석 화면은 요청 중 중복 제출을 방지하고, 실패 시 기존 사진으로 재시도할 수 있게 합니다.
