# 얼굴 Crop 확인 후 추론

## 기존 구조 분석

frontend/js/app.js의 selectPhoto는 업로드와 카메라 입력을 모두 받았고 원본 Blob으로 미리보기를 만들었습니다. 결과 버튼은 이 원본을 model.js를 통해 /api/predict로 보냈습니다. serve.py(8766)와 production.py(8767)는 동일한 local_model.ModelRegistry를 호출합니다. 모델은 시작 시 로드하며 FaceScorer.predict에서 EXIF/RGB 및 모델별 전처리 후 추론합니다.

## 기존 Haar 코드 재사용

cropping/crop_local_idol_faces_haar.py의 largest_face와 square_box를 직접 import합니다. 원본 파일은 수정하지 않았습니다.
기존 검출: BGR→gray, equalizeHist, detectMultiScale(scaleFactor=1.05, minNeighbors=5, minSize=(40,40)).
웹 입력은 PIL EXIF 보정/RGB로 읽으므로 RGB→gray를 사용하며 나머지 파라미터는 같습니다.
여러 얼굴은 면적이 가장 큰 하나를 고릅니다. margin=0.25, side=max(w,h)*1.5인 정사각형을 이미지 경계 안으로 이동/제한합니다.
기존의 224px 강제 축소와 JPEG 95 저장은 웹에서는 사용하지 않습니다. 원래 Crop 해상도의 무손실 PNG를 메모리에서 반환합니다.
기존 배치 코드는 읽기 실패/미검출을 건너뜁니다. 웹은 미검출 HTTP 422 JSON, 손상 이미지 HTTP 400 JSON으로 반환하며 원본을 추론하지 않습니다.
Haar XML은 OpenCV 패키지의 파일을 Python으로 읽어 FileStorage MEMORY로 로드합니다. 기존 Public 폴더 복사 방식 대신 메모리 읽기로 한글 Windows 경로 문제를 피합니다.

## API 계약

POST /api/crop:
- Content-Type: application/octet-stream, X-Filename: URL 인코딩된 원본 파일명
- JPG/JPEG/PNG/WebP 확장자, 실제 이미지 decode/포맷, 10MB 및 2,000만 화소 검증
- 성공: image/png 본문, X-Crop-Token, X-Face-Count 헤더
- 이 단계에서는 모델을 호출하지 않습니다.

POST /api/predict?model=efficientnet_b0:
- 본문: /api/crop이 반환한 정확히 같은 PNG 바이트
- X-Filename: face-crop.png, X-Crop-Token: Crop 응답 토큰
- ModelRegistry.predict_crop에서 HMAC 바이트 일치/30분 만료를 검증한 다음 기존 predict를 호출
- 원본 재전송, 다른 이미지, 토큰 누락/만료는 HTTP 409로 차단
- 얼굴 검출을 다시 실행하지 않습니다.

방법 A: 브라우저가 Crop Blob을 보관합니다. DB/영구 파일 저장은 없습니다. 토큰은 서버의 임의 메모리 키로 서명하므로 서버 재시작 후 사진을 다시 선택해야 합니다. 여러 서버 프로세스를 병렬 운영하는 구조는 지원하지 않습니다.

## 프런트엔드 코드 위치

- app.js/selectPhoto: 상태 초기화 → cropImage 호출 → Crop Blob만 photo.src에 연결
- model.js/cropImage: /api/crop 요청 및 PNG/토큰 반환
- app.js의 analyze 클릭 이벤트: 현재 selectedBlob과 cropToken을 predictImage로 전달
- app.js/clearPhoto: 이전 Blob/URL/점수/메시지/요청 초기화와 버튼 비활성화
- selectPhoto의 version 비교와 AbortController: 늦은 A 응답이 B를 덮어쓰지 않도록 처리
- 추론 결과는 version과 modelVersion을 함께 검사해 사진/모델 변경 이전의 결과를 무시
- camera capture도 selectPhoto 호출. 캔버스는 PNG로 생성해 JPEG 반복 압축을 피함
- index.html의 crop-help: 업로드 전/후·성공/실패와 무관하게 항상 표시

미리보기의 Blob과 추론 API 본문은 동일 객체/동일 바이트입니다. 이후 모델에 필요한 Resize 256 bicubic → CenterCrop 224 → ImageNet normalization은 유지합니다. 따라서 미리보기는 확인할 원본 얼굴 Crop이며, 정규화된 224×224 텐서 자체를 표시하는 것은 아닙니다.

## 파일별 변경

- face_crop.py: 공통 Crop/validation/토큰 서비스 추가
- local_model.py: CropService를 서버 시작 시 생성, predict_crop 검증 진입점 추가
- serve.py / production.py: Crop API 추가, Prediction API는 확인된 PNG만 허용
- frontend/js/model.js: Crop 요청 분리 및 토큰 포함 Prediction 요청
- frontend/js/app.js: 즉시 Crop, 버튼 추론, 상태 초기화·race 처리·동일 Blob 저장
- frontend/index.html / css/styles.css: 결과 보기 버튼·항상 표시되는 작은 안내 문구
- requirements*.txt: Haar XML이 포함된 opencv-python-headless==4.13.0.92
- test_crop.py / test_local.py / test_production.py / test_frontend.cjs: 아래 검증

## 테스트 결과

Python unittest 18개 통과:
- 실제 얼굴 검출/Crop과 원본 bounding box 픽셀 일치
- 넓은 배경 속 작은 얼굴 검출
- 서로 다른 크기의 실제 얼굴 2개를 합성한 이미지에서 가장 큰 얼굴 선택
- 얼굴 없는 이미지 422, 잘못된 파일 400
- 카메라 형태의 PNG 처리, 변조/만료 토큰 차단
- Crop 호출에서 모델 호출 0회, 원본 Prediction 차단
- 세 모델의 확인된 Crop 추론, API/직접 추론 값 일치
- 운영 Host/Origin·업로드 크기·정적 파일 접근 제한 유지

Node 상태 테스트 통과:
- 클릭 이전 추론 0회, 미리보기 Blob===Prediction Blob
- 새 사진 이전 점수/Crop 제거, 늦은 A 응답 무시
- 카메라가 공통 Crop 경로 사용, 미검출/잘못된 입력 시 버튼 차단
- Crop 대기 중 모델 변경으로 응답이 유실되지 않음

실물 카메라 촬영과 실제 브라우저 시각 검증은 자동화 도구 실행 오류로 수행하지 못했습니다. 카메라 콜백/PNG/API는 코드 실행 테스트로 검증했습니다.

실행:
```powershell
cd C:\sungwon\image_attractive_visionProject
.\face_score_web\.venv\Scripts\python.exe -m unittest discover -s face_score_web -p 'test_*.py'
node .\face_score_web\test_frontend.cjs
```

## 최종 흐름

사용자
├─ 파일 업로드
└─ 카메라 촬영
    ↓
원본 이미지 → 입력 검증
    ↓
Haar Cascade → 가장 큰 얼굴 Bounding Box
    ↓
25% 여백 정사각형 Face Crop → 무손실 PNG
    ↓
Crop 이미지 미리보기 (아직 추론하지 않음)
    ├─ Crop 이상 → 다른 사진 선택 → 상태 초기화
    └─ Crop 정상 → 사용자가 결과 보기 클릭
                     ↓
              동일 PNG + 확인 토큰 검증
                     ↓
              EfficientNet-B0 전처리
                     ↓
                 Regression
                     ↓
                 1~5 Score
