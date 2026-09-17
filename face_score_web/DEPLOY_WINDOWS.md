# Windows + Cloudflare Tunnel 배포

## 현재 구조와 실행 흐름

- 프로젝트 루트: 학습 데이터 images/, 라벨 CSV/XLSX, DenseNet121/, EfficientNet-B0/, MobileNet 학습 노트북, classical_cv/, PPT/.
- 실제 웹 실행 범위: face_score_web/frontend/, local_model.py, models/와 서버 진입점.
- frontend/index.html + css/styles.css: 사진 선택, 촬영, 모델 콤보박스, 점수 표시.
- frontend/js/app.js: 사진 미리보기와 선택 상태, model.js: 같은 출처 /api/predict 호출. localhost API 하드코딩 없음.
- serve.py: 기존 로컬 전용 표준 라이브러리 서버, 127.0.0.1:8766. Host/Origin이 localhost만 허용되어 인터넷 배포에는 사용하지 않음.
- production.py: 추가한 Flask/Waitress 진입점. 같은 화면 및 기존 ModelRegistry 재사용, 기본 127.0.0.1:8767.
- local_model.py: 앱 시작 시 모델을 한 번씩 로드. eval + inference_mode, RGB/크기 조정/정규화 후 예측.
- models/<모델>/config.json: 체크포인트 이름, 아키텍처, 전처리 및 점수 변환 계약.
- dist/: 이전 정적 배포 산출물. 이 배포에서 사용하지 않음.
- 추론에는 학습 데이터/라벨/노트북이 필요하지 않지만 DenseNet 빌더 import 때문에 DenseNet121/training/model.py 및 패키지 파일은 필요함.

외부 브라우저 HTTPS → Cloudflare → cloudflared → 127.0.0.1:8767 → Waitress/Flask → 기존 추론 → 1~5 JSON → 화면.
화면과 API가 동일 출처라 CORS 허용 설정과 프런트엔드 API 환경변수는 필요하지 않음. API 상대 경로를 유지함.

## 변경 파일과 이유

| 파일 | 변경 | 이유 |
|---|---|---|
| production.py | Waitress/Flask 운영 진입점, Host/Origin 검증, 파일 제공 범위 제한, 업로드 검증, 동시 추론 1건 제한 | Windows에서 외부 요청 처리; CPU/메모리 사용 제한 |
| local_model.py | 활성 모델 목록 인자 추가 | 작은 서버에서는 EfficientNet만 로드 가능 |
| frontend/js/model.js | X-Filename 헤더 추가 | 서버에서 확장자 검증; 한글 파일명 URL 인코딩 |
| frontend/js/app.js | 서버의 default_model 사용, 파일명 전달 | 운영 기본 EfficientNet 선택 |
| frontend/index.html | 사진 처리 문구 수정 | 모바일 사진은 Windows 서버로 전송됨을 반영 |
| requirements.txt | CPU PyTorch + Flask + Waitress 버전 지정 | 운영 환경 재현 |
| .env.example | 환경변수 예제 | 실제 환경설정은 Git에 저장하지 않음 |
| start_production.ps1 | 운영 실행 스크립트 | 동일한 Windows 실행 방법 제공 |
| cloudflared.example.yml | 고정 도메인 터널 예제 | 실제 자격증명 파일은 사용자 폴더에 저장 |
| 두 .gitignore | 모델/환경/인증서/임시파일 제외 보강 | 불필요한 신규 Git 업로드 방지 |
| test_production.py, test_local.py | 운영 계약 검증, 변경된 함수 호출 검사 | 기존 기능과 공개 요청 검증 |

## 모델과 업로드

현재 EfficientNet 가중치 파일명은 models/efficientnet_b0/EfficientNetAllImage.pt 이다. best.pt로 바꾸려면 해당 폴더에 넣고 config.json의 checkpoint만 best.pt로 수정한 후 재시작한다. 단, 아키텍처·전처리·출력 계약이 동일해야 한다.
EfficientNet: Resize 256 bicubic → CenterCrop 224 → ImageNet Normalize → 1 + 4*sigmoid.
기본 3종 모델을 시작 시 한 번씩 로드하며 요청마다 재로드하지 않는다. 기본 선택은 efficientnet_b0.
ENABLED_MODELS=efficientnet_b0이면 이 모델만 로드한다. 모델 교체 후 프로세스를 다시 시작한다.
JPG/JPEG/PNG/WebP 확장자와 실제 이미지 포맷, 최대 10MB/2,000만 화소를 검사한다.
사진은 파일로 저장하지 않는다. Waitress 요청 버퍼도 허용 업로드 크기까지 메모리에 유지한다. 한 번에 한 건 추론하고 추가 동시 요청은 429로 재시도를 안내한다.
로그인 없는 공개 서비스이므로 주소를 가진 사용자가 요청할 수 있다. Host/Origin 검사는 사용자 인증이 아니다.

## PowerShell: 처음부터 실행

다음은 프로젝트가 이미 있는 Windows PC 기준. Python 3.12가 필요하다.

### 1. 준비 (PowerShell 창 A)

```powershell
cd C:\sungwon\image_attractive_visionProject\face_score_web
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test_production.py
```

이 테스트는 test_photo_9/F_2.png와 세 모델 파일이 있는 현재 프로젝트 기준이다. 웹 운영에는 테스트 데이터가 필요하지 않다.

### 2. cloudflared 설치 (PowerShell 창 B)

winget이 있는 PC:

```powershell
winget install --id Cloudflare.cloudflared --exact
```

설치 후 PowerShell을 새로 열어 `cloudflared --version` 확인.
winget이 없는 PC에서는 https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/ 의 Windows 64-bit 실행 파일을 내려받아 C:\Tools\cloudflared.exe 에 놓고 아래 cloudflared 명령 대신 `C:\Tools\cloudflared.exe`를 사용한다.

### 3. 운영 앱 로컬 기동 (창 A)

```powershell
$env:HOST = '127.0.0.1'
$env:PORT = '8767'
$env:DEFAULT_MODEL = 'efficientnet_b0'
$env:ENABLED_MODELS = 'densenet121,mobilenetv3,efficientnet_b0'
$env:PUBLIC_ORIGIN = ''
.\.venv\Scripts\python.exe production.py
```

http://localhost:8767 에서 화면과 점수 확인. 기존 8766 로컬 서버와 별도로 실행한다.
.env.example은 참고용이며 자동 로드하지 않는다. PowerShell의 $env: 변수로 지정한다.

### 4. 임시 HTTPS 터널 생성 (창 B)

```powershell
cloudflared tunnel --url http://127.0.0.1:8767
```

창에 출력되는 https://무작위이름.trycloudflare.com 주소를 복사한다. 이 창을 유지한다.
현재 단계에서 외부 접속이 403이면 다음 단계의 허용 주소 설정이 필요한 상태다.

### 5. 공개 주소 허용 (창 A)

앱 창에서 Ctrl+C로 종료하고 아래 주소를 실제 출력된 주소로 바꾼다.

```powershell
$env:PUBLIC_ORIGIN = 'https://실제로-발급된-이름.trycloudflare.com'
.\.venv\Scripts\python.exe production.py
```

PUBLIC_ORIGIN에는 경로 없이 HTTPS 주소만 넣는다. Tunnel 재시작으로 주소가 바뀌면 이 값도 변경하고 앱을 재시작한다.
Quick Tunnel은 테스트/발표용이며 주소가 바뀌고 가동 보장이 없다. 계속 운영하려면 아래 고정 도메인 Tunnel을 쓴다.

## 고정 도메인으로 운영

Cloudflare 계정과 Cloudflare DNS에서 관리하는 도메인이 필요하다. 별도 Windows HTTPS 인증서 설치는 필요 없다.
대시보드에서 Tunnel 생성 → Windows 커넥터 설치 안내 실행 → 공개 hostname face.example.com 설정 → 서비스 http://127.0.0.1:8767 지정.
앱은 PUBLIC_ORIGIN=https://face.example.com 으로 실행한다. HTTP Host Header override는 설정하지 않고 원래 공개 Host를 전달한다.
발급되는 Tunnel 토큰은 비밀이며 Git이나 이 문서에 붙여 넣지 않는다.

CLI로 관리하려면 (도메인과 UUID는 실제 값으로 변경):

```powershell
cloudflared tunnel login
cloudflared tunnel create face-score
cloudflared tunnel route dns face-score face.example.com
cd C:\sungwon\image_attractive_visionProject\face_score_web
Copy-Item cloudflared.example.yml cloudflared.local.yml
notepad cloudflared.local.yml
# tunnel UUID, 사용자 경로, hostname 수정 후 저장
cloudflared tunnel --config .\cloudflared.local.yml run face-score
```

앱과 Tunnel 프로세스가 모두 실행 중이어야 한다. 장기 운영은 Cloudflare의 Windows 서비스 및 Windows 작업 스케줄러로 앱 자동 시작을 구성한다. 현재 작업에서는 자동 부팅 서비스는 설치하지 않았다.

## Windows 네트워크

같은 PC에서 실행되는 Tunnel에는 HOST=127.0.0.1이 적합하다. 공유기 포트포워딩 및 Windows 8767 인바운드 허용은 필요 없다.
cloudflared가 외부로 나가는 TCP/UDP 7844와 DNS 통신을 사용할 수 있어야 한다. 기관망이 차단하면 관리자에게 아웃바운드 허용 요청. UDP가 막히고 TCP는 허용되면 `cloudflared tunnel --protocol http2 --url http://127.0.0.1:8767`을 사용한다.
필요한 경우 HOST=0.0.0.0으로 지정 가능하지만 Tunnel 방식에는 불필요하다. 공인 주소의 허용 여부는 PUBLIC_ORIGIN으로 별도 통제한다.
PC 절전/종료, 인터넷 단절, 앱/Tunnel 종료 시 외부 서비스도 중단된다.

## Git 점검 결과

현재 인덱스에서 .pt 모델은 추적되지 않는다. 새 .env.*, 인증서, 터널 자격증명 폴더, 로그/임시파일도 제외한다. 공유 가능한 .env.example과 cloudflared.example.yml은 예외로 추적 가능하다.
하지만 images 5,182장, test_photo_9 9장, labels_combined_1to5.csv 및 .xlsx는 이미 추적 중이다. .gitignore는 기존 추적/과거 커밋을 지우지 않는다. 과거 원격 공개 여부 및 이력 전체는 이번에 조사하지 않았다.
이 데이터는 웹 배포에 필요하지 않다. 안전한 배포 저장소에는 웹 소스/설정 예제와 DenseNet 모델 정의만 복사하고, 모델은 별도 전달하는 방식이 적합하다.
현재 저장소에서 추적만 해제하려면 아래 명령을 검토 후 실행한다. 로컬 원본은 남지만 다음 커밋에는 Git 삭제로 나타나며 과거 이력은 그대로다. 이번 작업에서는 실행하지 않았다.

```powershell
cd C:\sungwon\image_attractive_visionProject
# 사전 확인
 git ls-files -- images test_photo_9 labels_combined_1to5.csv labels_combined_1to5.xlsx
# 선택적으로 인덱스에서만 제외
 git rm -r --cached -- images test_photo_9
 git rm --cached -- labels_combined_1to5.csv labels_combined_1to5.xlsx
 git diff --cached --stat
```

## 외부 테스트

1. 스마트폰 Wi-Fi를 끄고 LTE/5G로 발급된 HTTPS URL에 접속.
2. /api/health에서 ready=true 및 default_model=efficientnet_b0 확인.
3. 한글 이름의 정상 JPG/PNG 업로드 → 점수 확인 → 1~5 점수, EfficientNet-B0 이름 확인.
4. 동일 이미지를 로컬과 외부에서 선택해 동일 모델 점수가 같은지 비교.
5. 세 모델 전환 및 카메라 권한/촬영 테스트.
6. 잘못된 확장자/손상 이미지/10MB 초과 업로드 거절 확인. 동시에 분석하면 다른 요청에 잠시 후 재시도 안내.
7. /models/efficientnet_b0/EfficientNetAllImage.pt 및 /.env 는 404여야 함.
8. 403: PUBLIC_ORIGIN 확인. 502: Windows 앱/포트 확인. 429: 추론 완료 후 재시도.

## 무료 클라우드 검토 (2026-09-17 확인)

- Render Python Web Service: Docker 없이 가능한 구조. requirements.txt, HOST=0.0.0.0, 플랫폼 PORT, PUBLIC_ORIGIN을 지정하고 python face_score_web/production.py 실행. 모델은 Git 대신 비공개 저장소/스토리지에서 안전하게 내려받는 배포 절차가 추가로 필요하다. 무료 512MB/0.1CPU, 15분 유휴 정지 때문에 PyTorch 3종 상시 로딩에는 여유가 작다. 현재 Windows 3종 로드 직후 working set 약 377MiB이며 운영 OS의 RSS와 동일한 지표가 아니므로 무료 플랜 적합 판정으로 사용할 수 없다. EfficientNet 단독 활성화 후 Linux에서 메모리/속도 검증이 필요하다.
- Hugging Face Spaces: 현재 공식 문서상 일반 Gradio/Docker compute Space 생성에는 유료 플랜이 필요하다. 적격 무료 개인 계정의 ZeroGPU Gradio 예외는 있지만 현재 Flask/Waitress를 그대로 올리는 경로가 아니므로 우선 추천하지 않는다.
- 정적 호스팅만으로는 .pt PyTorch 추론을 실행할 수 없다.

현재 조건에서는 Windows + Cloudflare Tunnel이 코드 변경과 모델 배포 부담이 가장 작다.

## 검증 범위

기존 통합 7개 + 운영 테스트 5개 통과. 실제 Waitress health 및 실제 EfficientNet 추론 확인. 외부 Host/Origin은 테스트 클라이언트로 확인.
cloudflared 설치/로그인/터널 발급 및 스마트폰 실접속은 아직 수행하지 않았다.

공식 참고:
- https://flask.palletsprojects.com/en/stable/deploying/waitress/
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/
- https://developers.cloudflare.com/tunnel/configuration/
- https://render.com/docs/compute-plans
- https://render.com/docs/free
- https://huggingface.co/docs/hub/spaces-overview
