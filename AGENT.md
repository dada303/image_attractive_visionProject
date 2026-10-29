# 백엔드 구조 및 개발 가이드

## 목적과 적용 범위

이 문서는 `face_score_web_add_cropping/backend`의 계층별 작업 기준이다. 초기 제안을 바탕으로 기존 구현을 이동했으며, 코드 증가를 최소화하기 위해 아래 실제 구조를 사용한다. 뒤의 상세 목표 구조는 초기 제안으로, 추가 파일을 반드시 만들라는 지시가 아니다.

## 현재 적용된 구조

```text
backend/
├── production.py                 # Uvicorn 운영 실행, 기존 create_app import 유지
├── serve.py                      # Uvicorn 로컬 실행
├── app.py                        # FastAPI 설정 검증·앱 조립
├── config.py                     # 공통 경로·처리 제한값
├── controllers/
│   └── image_controller.py       # 공통 FastAPI API·정적 파일·HTTP 정책
├── services/
│   ├── crop_service.py           # 기존 CropService
│   └── model_service.py          # 기존 ModelRegistry
├── repositories/
│   └── model_repository.py       # 기존 설정·체크포인트 읽기 추출
├── inference/
│   └── face_scorer.py            # 기존 FaceScorer
└── common/
    ├── errors.py                 # 기존 CropError
    └── request_validation.py     # 기존 validate_upload
```

MVC의 View는 기존 `frontend/`, Controller는 FastAPI 기반 `controllers/`, Model 역할은 서비스·저장소·추론 계층이 담당한다. DB가 없고 현재 dict·tuple 결과로 충분하므로 `entities/`와 새 데이터 클래스를 추가하지 않았다. Crop 알고리즘은 기존 서비스 안에 유지했고 별도 검출기 래퍼를 만들지 않았다. 기존 기능·반환값을 유지하며 각 모듈·클래스·함수에 역할 설명을 둔다.

Python 관례에 따라 폴더명은 소문자 복수형, 파일명은 `snake_case`를 사용한다. 핵심 흐름은 `Controller → Service → Repository`이며, Entity는 각 계층에서 사용하는 도메인 데이터를 표현한다. 모든 요청이 Repository까지 거쳐야 하는 것은 아니다.

## 이관 전 구조와 이동 기준 (이력)

| 이관 전 파일 | 이관 전 책임 | 분리 방향 |
| --- | --- | --- |
| `backend/production.py` | Flask 생성, 환경 설정, 보안 검사, 라우팅, 동시 처리 제한, 응답 변환, 운영 실행 | 이후 FastAPI·Uvicorn의 공통 컨트롤러·서비스 조립으로 이관 |
| `backend/serve.py` | 로컬 HTTP 서버, 요청 검사, API 처리, 정적 파일 제공 | 로컬 진입점은 유지하고 동일한 서비스를 호출 |
| `backend/local_model.py` | 설정·체크포인트 읽기, 신경망 생성, 전처리, 추론, 모델 목록, Crop 검증 연결 | Repository·Inference·Service로 책임별 분리 |
| `backend/face_crop.py` | 업로드 검증, 얼굴 검출, Crop 생성, 토큰 발급·검증 | HTTP 입력 검증·Crop 서비스·영상 처리 구현으로 분리 |

현재 DB와 ORM은 사용하지 않는다. 업로드 이미지는 메모리에서 처리하며 파일로 저장하지 않는다. 폴더 분리만을 위해 DB나 ORM을 추가하지 않는다. 루트 `models/`의 학습 코드와 웹 앱 하위 `models/`의 배포용 설정·가중치도 구별한다.

## 초기 권장안 (이력)

아래는 실제 적용 전의 책임 분리 제안이다. 현재 적용 구조는 위 `현재 적용된 구조`를 기준으로 하고, 이 제안의 파일을 새로 만들지 않는다.

```text
face_score_web_add_cropping/
├── backend/
│   ├── __init__.py
│   ├── production.py                 # 운영 실행 및 create_app 호환 진입점
│   ├── serve.py                      # 로컬 HTTP 서버 진입점
│   ├── app.py                        # 당시 제안한 앱 생성·계층 조립
│   ├── config.py                     # 환경 변수, 공통 제한값, 기준 경로
│   ├── controllers/
│   │   ├── image_controller.py        # /api/crop, /api/predict
│   │   ├── model_controller.py        # /api/health, /api/models
│   │   ├── static_controller.py       # 프런트엔드 정적 파일 제공
│   │   └── local_handler.py           # SimpleHTTPRequestHandler 어댑터
│   ├── services/
│   │   ├── crop_service.py            # Crop 처리와 확인 토큰 발급·검증
│   │   ├── prediction_service.py      # 토큰 검증 후 모델 선택·추론 조정
│   │   └── model_service.py           # 모델 초기화, 가용 상태·목록 관리
│   ├── entities/
│   │   ├── model_config.py            # 모델 설정 데이터
│   │   ├── crop_result.py             # PNG, 토큰, 얼굴 정보
│   │   └── prediction_result.py       # 점수와 모델 정보
│   ├── repositories/
│   │   └── model_repository.py        # config.json·체크포인트 읽기
│   ├── inference/
│   │   ├── face_scorer.py             # PyTorch 모델 생성·전처리·추론
│   │   └── face_detector.py           # OpenCV 얼굴 검출·Crop 계산
│   └── common/
│       ├── errors.py                  # 공통 예외
│       ├── request_validation.py      # 파일명·Content-Type·길이 검사
│       └── http_policy.py             # Host·Origin·보안 헤더·오류 응답 연결
├── models/                           # 기존 배포용 설정·가중치 위치 유지
├── frontend/                         # 기존 프런트엔드 위치 유지
└── tests/
    ├── test_controllers.py
    ├── test_services.py
    ├── test_model_repository.py
    └── test_inference.py
```

## 계층별 책임과 금지 사항

### Entity

- 업무 데이터를 표현한다. 필요하면 표준 라이브러리의 `dataclass`를 사용한다.
- 현재 프로젝트에서는 ORM Entity가 아니라 설정·처리 결과 같은 도메인 데이터 객체다. 고유 식별자와 생명주기를 갖는 영속 Entity는 저장 요구가 생길 때 도입한다.
- FastAPI 요청·응답 객체, 파일 읽기, 모델 로딩, GPU 초기화를 포함하지 않는다.
- PyTorch 신경망을 Entity에 넣지 않는다. 머신러닝 모델은 `inference/`의 책임이다.
- 내부 객체를 도입하더라도 외부 JSON 필드명과 값의 의미는 유지한다.

### Controller

- URL, HTTP 메서드, 헤더·쿼리·본문 입력, HTTP 상태 코드, JSON·PNG 응답 변환을 담당한다.
- 업로드의 파일명·Content-Type·요청 길이를 검사한 뒤 서비스에 일반 Python 값으로 전달한다.
- 서비스 결과와 예외를 기존 API 형식으로 변환한다.
- 직접 체크포인트를 읽거나 신경망 추론·얼굴 검출·토큰 서명 로직을 구현하지 않는다.
- FastAPI 컨트롤러는 로컬·운영 서버가 함께 사용한다. URL·요청 헤더·응답 변환만 담당한다.

### Service

- Crop 생성, 확인 토큰 검증, 사용할 모델 선택, 추론 실행 순서 같은 업무 흐름을 담당한다.
- `PredictionService`는 Crop 토큰을 먼저 검증하고 추론을 호출한다. 예측 시 얼굴을 다시 검출하지 않는다.
- 이미지 포맷·화소 수 같은 내용 검증은 영상 처리 모듈과 협력한다. HTTP 입력 검증만으로 이를 대체하지 않는다.
- FastAPI의 `Request`, `Response`, `JSONResponse`를 import하지 않는다.
- Repository와 추론 객체를 생성자 등으로 전달받는다. 운영 동시 처리 제한은 공유 서비스 인스턴스 또는 앱에서 조립하는 공통 실행 제어에 두고 요청마다 새로 만들지 않는다.

### Repository

- 외부 저장소의 데이터를 읽고 쓰는 경계다. 현재는 모델별 `config.json`과 `.pt` 체크포인트 읽기가 해당한다.
- 설정 파싱·체크포인트 역직렬화와 그 과정의 파일 오류를 담당한다. 체크포인트 로딩의 `weights_only=True` 등 기존 안전 설정을 유지한다.
- 신경망 생성, GPU 이동, 전처리, 추론, 점수 보정, HTTP 응답을 담당하지 않는다.
- 학습 가중치를 읽는 역할이므로 `ModelRepository`처럼 대상을 명확히 명명한다. `ModelRegistry` 전체를 이름만 바꿔 옮기지 않는다.
- 실제 저장 기능이 없는 이미지·예측 결과용 Repository는 만들지 않는다. 범용 CRUD 기반 클래스나 인터페이스도 필요가 확인되기 전에는 추가하지 않는다.

### Inference 및 공통 모듈

- `inference/`는 PyTorch·OpenCV 구현을 모은다. Repository가 읽은 설정·가중치로 모델을 구성하고 추론하며, 필요한 잠금을 유지한다.
- `common/`에는 여러 곳에서 실제로 공유하는 작은 기능만 둔다. 업무 로직을 모으는 잡동사니 폴더로 사용하지 않는다.
- 기존 `CropError.status`는 단계적 이관 중 유지할 수 있다. HTTP와 무관한 예외로 변경한다면 컨트롤러에서 동일한 상태 코드로 매핑하는 검증을 먼저 마련한다.

## 의존성과 인스턴스 수명

```text
진입점 / app.py → 설정 로딩 및 객체 조립
Controller → Service → Repository
                    → Inference
각 계층 → 필요한 Entity
```

Repository와 Inference는 Controller·Service를 역으로 import하지 않는다. Entity는 다른 계층에 의존하지 않는다. 서비스는 순환 의존하지 않도록 `PredictionService → CropService / ModelService` 방향으로 연결한다.

모델, 얼굴 검출기, 토큰 서명 비밀값, 잠금은 프로세스 시작 시 생성해 재사용한다. 요청마다 생성하면 성능 저하뿐 아니라 이전 Crop 토큰 검증 실패가 발생할 수 있다. 기존 운영 동시 요청 제한과 모델·검출기 잠금의 역할을 구분해 보존한다.

FastAPI 컨트롤러는 원본 이미지 바이너리를 그대로 읽어 서비스에 전달한다. `multipart/form-data`나 파일 저장으로 바꾸지 않으며, 10MB 제한·30초 업로드 시간 제한·`application/octet-stream` 계약을 보존한다. 동기 PyTorch·OpenCV 작업은 이벤트 루프를 막지 않도록 스레드 풀에서 실행하되, 기존 1개 처리 슬롯을 유지한다.

Uvicorn은 단일 worker로 실행한다. worker를 늘리면 worker마다 모델, 처리 슬롯, Crop 토큰 서명 비밀값이 따로 생성되어 Crop 응답과 다음 예측 요청이 다른 worker에 도달할 때 토큰 검증이 실패할 수 있다. Cloudflare Tunnel에서 전달하는 요청도 기존 Host·Origin 정책으로 검증하며, 프록시 헤더를 신뢰하도록 바꾸지 않는다.

## 점진적 이관 순서

1. 현재 변경 사항과 테스트 상태를 확인하고 기존 사용자 작업을 보존한다. 실행을 막는 기존 오류가 있다면 구조 변경과 구별해 기록하고 별도로 처리한다.
2. 계층 이동 전에 현재 API 계약을 회귀 테스트로 고정한다. 모델 가중치 없이 사용할 수 있는 가짜 Registry·추론 객체를 우선 활용한다.
3. 공통 설정·예외·입력 검증을 추출한다. 서로 중복된 제한값은 한 곳에서 정의한다.
4. `local_model.py`에서 설정·체크포인트 읽기를 Repository로, 모델 생성·추론을 Inference로 분리한다.
5. `ModelRegistry`의 모델 관리와 예측 흐름을 Service로 분리하고 Crop 처리도 서비스 경계에 맞춰 정리한다.
6. 당시 Flask 라우트와 로컬 핸들러를 Controller로 추출하고 같은 Service에 연결했다. 이후 FastAPI 공통 컨트롤러로 통합했다.
7. `app.py`에서 의존성을 조립한다. 기존 `python -m backend.serve`, `python -m backend.production`, `create_app(registry=...)` 사용 경로는 호환 진입점으로 유지한다.
8. 호출부·테스트·실행 스크립트·문서의 참조를 확인한 후에만 기존 모듈을 제거한다. 변경된 파일 깊이 때문에 `Path(__file__).parents[...]`의 의미가 바뀌므로 모델·프런트엔드·학습 모듈 경로도 확인한다.

단순히 파일을 계층 폴더로 옮기는 것으로 끝내지 않는다. 한 파일에 섞인 책임을 먼저 나누고, 하나의 책임 단위로 이동·검증한다. 폴더 분리 목적으로 새로운 프레임워크나 의존성을 추가하지 않는다.

## 완료 기준과 검증

- `/api/health`, `/api/models`, `/api/crop`, `/api/predict`의 기존 요청·응답 계약을 FastAPI 공통 서버에서 유지한다.
- Crop 응답의 PNG 본문, `X-Crop-Token`, `X-Face-Count`와 예측 JSON 필드를 유지한다.
- 확장자·Content-Type·10MB 크기 제한·2,000만 화소 제한·얼굴 미검출·모델 미가용·잘못된 모델을 검증한다.
- 토큰과 이미지의 결합, 이미지 변조 거부, 1,800초 만료, 잘못된 토큰 거부를 검증한다.
- Host·Origin 검사, 보안 헤더, 운영 동시 요청 거부와 예외 발생 후 슬롯 반환을 검증한다.
- 단위 테스트는 Service에 가짜 의존성을 주입해 수행한다. FastAPI 테스트 클라이언트로 공통 HTTP 계약을 확인한다.
- 실제 가중치가 있는 환경에서는 모델 로딩·추론 스모크 테스트와 두 실행 경로를 확인한다. 업로드 이미지를 디스크에 저장하지 않는 기존 동작을 유지한다.
- 프로젝트에 구성된 lint·정적 분석·테스트를 실행하고 Python 문법·import도 검사한다. 실제 가중치나 실행 환경이 없어 확인하지 못한 항목은 완료 보고에 명시한다.

## 문서 파일명

이 파일은 요청한 이름인 `AGENT.md`로 작성했다. Codex의 자동 지침 탐색에 사용하는 이름은 `AGENTS.md`이므로, 자동 적용이 필요할 때는 이 내용을 루트 `AGENTS.md`에 반영한다. 기존 지침 파일이 있으면 덮어쓰지 않고 병합한다.
