# 아키텍처

브라우저(파일 선택 / 카메라 촬영)
→ 로컬 미리보기 및 분석 버튼
→ POST /api/v1/predict (multipart/form-data)
→ 업로드 검증 → 이미지 디코딩 / EXIF 방향 보정 / RGB
→ 학습과 동일한 전처리 → DenseNet121 eval + inference_mode
→ 학습과 동일한 출력 변환 → 점수 JSON
→ 미리보기 아래 점수 / 모델 버전 / 다시 분석

학습은 Colab 등 별도 환경에서 수행하고 웹 서버에는 확정된 추론 모델만 배포합니다.
기본은 촬영 버튼으로 정지 이미지 한 장을 보내는 방식입니다. 실시간 연속 추론은 후속 범위입니다.

## 구성요소와 구현 예정 파일
frontend/index.html: 파일/카메라 탭, 미리보기, 분석 상태, 결과 카드
frontend/css/styles.css: 데스크톱/모바일 스타일
frontend/js/app.js: 상태 전환(idle, preview, analyzing, result, error)
frontend/js/camera.js: getUserMedia, 캡처 Blob, 카메라 track 종료
frontend/js/api.js: 요청, 타임아웃, 오류 메시지
frontend/js/result.js: 결과 표시, 필요 시 브라우저에서 PNG 다운로드

backend/app/main.py: FastAPI lifespan과 정적 파일 연결
backend/app/api/predict.py: 요청 검증·추론 호출
backend/app/api/health.py: 생존 및 모델 준비 상태
backend/app/schemas/prediction.py: JSON 계약
backend/app/services/image_service.py: 크기 제한·실제 형식 확인·EXIF/RGB
backend/app/ml/model_loader.py: 모델 구조·가중치 strict 로딩·메타데이터 검증
backend/app/ml/preprocess.py: 학습 전처리 재현
backend/app/ml/predictor.py: eval/inference_mode 및 출력 점수 변환
backend/app/core/config.py: 모델 경로, 업로드 제한, 장치 설정

## 모델 실행
모델은 서버 프로세스 시작 때 한 번 로드합니다. 준비 전 readiness는 503입니다.
처음에는 서버 worker 1개, 동시 추론 1개로 시작하여 CPU/GPU 메모리 중복을 피합니다.
추론은 이벤트 루프를 막지 않도록 작업 스레드에서 수행하고 동시성 제한을 적용합니다.
CPU로 기능을 완성한 뒤 동일 환경에서 지연시간을 측정하여 GPU 필요 여부를 결정합니다.
가짜 점수나 임의의 기본 모델을 성공 응답으로 내보내지 않습니다.

## 얼굴 처리
DenseNet121 자체는 얼굴 위치를 찾는 모델이 아닙니다.
1차 버전은 사용자가 얼굴 한 명이 중심에 보이는 사진을 제출하는 흐름입니다.
학습에 얼굴 검출/정렬/crop이 있었다면 그 처리를 그대로 서버에 구현합니다.
학습에 없던 얼굴 crop을 임의로 추가하지 않고, 추가할 때는 기존 테스트셋으로 재평가합니다.
별도 검출기를 도입한 경우에만 NO_FACE/MULTIPLE_FACES를 객관적으로 반환할 수 있습니다.
검출기 없는 버전에서는 얼굴 수 자동 검증을 지원한다고 표시하지 않습니다.

## 입력 및 운영
카메라는 HTTPS 또는 localhost 및 사용자의 브라우저 권한이 필요합니다.
카메라 권한 거절 시 파일 업로드로 이어지고, 재촬영/화면 종료 시 track.stop()을 호출합니다.
초기 파일 형식은 JPEG/PNG/WebP, 한 장 최대 10 MiB / 20 MP를 제안합니다.
확장자와 MIME만 믿지 않고 디코딩한 형식·픽셀 수도 검사합니다.
한글 파일명은 표시용이며 서버 저장 경로로 사용하지 않습니다.
원본 사진은 기본적으로 영구 저장하지 않고 요청 처리 후 임시 리소스를 정리합니다.
로그에는 request_id, 모델 버전, 지연시간, 오류 코드만 남기고 사진 본문은 남기지 않습니다.
화면에는 데이터셋 라벨을 학습한 예측 점수라는 설명을 표시하고 성격·능력 같은 추가 의미를 부여하지 않습니다.
회귀값을 확률이나 정확도 퍼센트로 표시하지 않습니다.

## 배포
개발: localhost의 단일 FastAPI 서비스가 HTML과 API를 제공.
배포: HTTPS 역방향 프록시 → 동일 서비스. 서비스 환경 변수로 모델 버전과 위치 고정.
DB/외부 저장소/작업 큐는 사용자 기록·대량 처리 요구가 생길 때 추가합니다.

참고:
https://fastapi.tiangolo.com/tutorial/request-files/
https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
