# 모델 인계 폴더

다음 항목을 확정한 뒤 모델을 연결합니다. 현재 가중치는 포함되어 있지 않습니다.

- model.pt: PyTorch인 경우 학습한 state_dict 및 필요한 설정. 다른 프레임워크라면 원래 형식 유지.
- model_manifest.json: manifest.template.json의 null 항목을 실제 값으로 채움.
- 전처리: EXIF, RGB, resize 방식·크기, crop, mean/std, 얼굴 검출 여부.
- 출력: 회귀 1개인지 클래스 확률인지, 선형/sigmoid/scale/clipping 적용 위치.
- 구조: 실제 마지막 층 구성. DenseNet이라는 이유만으로 임의 헤드를 사용하지 않음.
- 라이브러리 버전, 체크포인트 SHA-256, 학습 도메인, 동일 조건 평가 결과.
- 기준 이미지의 예상 출력: tests에 실제 값으로 검증.

이전 EfficientNet 모델의 출력 변환을 DenseNet에 그대로 복사하지 않습니다.
모델 파일은 브라우저에 제공하지 않고 backend만 읽습니다.
