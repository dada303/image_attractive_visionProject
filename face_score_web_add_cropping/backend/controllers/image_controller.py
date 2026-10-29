"""FastAPI에서 기존 바이너리 API·정적 화면·오류 응답·보안 정책을 제공한다."""
import asyncio
import logging
from threading import BoundedSemaphore

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse, JSONResponse, Response

from ..config import FRONTEND, MAX_BYTES
from ..common.errors import CropError
from ..common.request_validation import validate_upload

SECURITY_POLICY = "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"


def register_routes(app, registry, hosts, origins, default_model, *, local=False):
    """모드별 호환 응답을 유지하며 하나의 HTTP 구현과 처리 슬롯을 등록한다."""
    inference_slot = BoundedSemaphore(1)
    logger = logging.getLogger(__name__)

    @app.middleware('http')
    async def validate_request(request, call_next):
        """Host·Origin을 검사하고 오류를 포함한 모든 응답에 보안 헤더를 붙인다."""
        if request.headers.get('host', '').lower() not in hosts:
            message = 'localhost에서 접속해주세요.' if local else '허용되지 않은 접속 주소입니다. PUBLIC_ORIGIN 설정을 확인해주세요.'
            response = JSONResponse({'error': message}, status_code=403)
        elif request.headers.get('origin') and request.headers['origin'] not in origins:
            message = 'localhost에서 접속해주세요.' if local else '허용되지 않은 요청 출처입니다.'
            response = JSONResponse({'error': message}, status_code=403)
        elif local and request.method == 'POST' and request.url.path not in ('/api/crop', '/api/predict'):
            response = JSONResponse({'error': '없는 API입니다.'}, status_code=404)
        elif not local and request.method == 'OPTIONS':
            # Flask의 자동 OPTIONS 응답은 유지하되 새 CORS 허용 정책을 추가하지 않는다.
            allowed = 'GET, HEAD, OPTIONS' + (', POST' if request.url.path in ('/api/crop', '/api/predict') else '')
            response = Response(headers={'Allow': allowed})
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception('Request failed')
                response = JSONResponse({'error': '요청을 처리할 수 없습니다.'}, status_code=500)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if not local:
            response.headers['Referrer-Policy'] = 'no-referrer'
            response.headers['Content-Security-Policy'] = SECURITY_POLICY
        elif request.url.path == '/docs':
            # Swagger의 CDN 자원과 초기화 스크립트는 개발 문서 페이지에서만 허용한다.
            response.headers['Content-Security-Policy'] = SECURITY_POLICY.replace(
                "style-src 'self'; script-src 'self'",
                "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'")
        return response

    @app.exception_handler(HTTPException)
    @app.exception_handler(RequestValidationError)
    async def http_error(request, error):
        """프레임워크 오류를 기존 error 필드로 변환하고 자동 422 응답을 방지한다."""
        status = error.status_code if isinstance(error, HTTPException) else 400
        messages = {413: '10MB 이하의 사진을 선택해주세요.', 404: '없는 경로입니다.', 400: '잘못된 요청입니다.'}
        return JSONResponse({'error': messages.get(status, '요청을 처리할 수 없습니다.')}, status_code=status,
                            headers=getattr(error, 'headers', None))

    @app.get('/api/health')
    @app.head('/api/health', include_in_schema=False)
    @app.get('/api/models')
    @app.head('/api/models', include_in_schema=False)
    def health():
        """기존 로컬·운영별 상태 JSON과 모델 목록을 반환한다."""
        result = {'ready': bool(registry.scorers) if local else True, 'models': registry.list_models()}
        if not local:
            result['default_model'] = default_model
        return JSONResponse(result)

    async def read_image(request, length):
        """본문을 최대 10MB·30초로 제한해 읽고 잘린 업로드를 거부한다."""
        data = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_BYTES:
                        raise CropError('10MB 이하의 사진을 선택해주세요.', 413)
                    data.extend(chunk)
        except TimeoutError as error:
            raise CropError('사진 전송 시간이 초과되었습니다.', 408) from error
        if len(data) != length:
            raise ValueError('사진 전송이 완료되지 않았습니다.' if local else '잘못된 요청입니다.')
        return bytes(data)

    # 프런트엔드와 동일한 바이너리 본문·헤더를 Swagger에서도 사용할 수 있게 명시한다.
    upload_schema = {
        'requestBody': {'required': True, 'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}}},
        'parameters': [
            {'name': 'X-Filename', 'in': 'header', 'required': True, 'schema': {'type': 'string'}},
            {'name': 'X-Crop-Token', 'in': 'header', 'schema': {'type': 'string'}},
            {'name': 'model', 'in': 'query', 'schema': {'type': 'string', 'default': default_model}},
        ],
    }

    @app.post('/api/crop', openapi_extra=upload_schema, responses={200: {'content': {'image/png': {}}}})
    @app.post('/api/predict', openapi_extra=upload_schema)
    async def process_image(request: Request):
        """업로드를 검사하고 처리 슬롯 안에서 기존 동기 서비스를 스레드로 실행한다."""
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            message = '잘못된 요청 크기입니다.' if local else '잘못된 요청입니다.'
            return JSONResponse({'error': message}, status_code=400)
        try:
            if local and not 0 < length <= MAX_BYTES:
                raise CropError('10MB 이하의 사진을 선택해주세요.', 413)
            validate_upload(request.headers.get('X-Filename'), request.headers.get('content-type'), length)
        except CropError as error:
            return JSONResponse({'error': str(error)}, status_code=error.status)
        if not inference_slot.acquire(blocking=False):
            return JSONResponse({'error': '다른 사진을 처리 중입니다. 잠시 후 다시 시도해주세요.'}, status_code=429)
        try:
            data = await read_image(request, length)
            if request.url.path == '/api/crop':
                png, token, metadata = await run_in_threadpool(registry.crops.prepare, data)
                return Response(png, media_type='image/png', headers={
                    'X-Crop-Token': token, 'X-Face-Count': str(metadata['face_count'])})
            # Flask는 첫 쿼리 값을 사용하며 로컬 서버는 빈 model 값을 기본값으로 처리했다.
            model_id = next((value for value in request.query_params.getlist('model') if not local or value), default_model)
            result = await run_in_threadpool(registry.predict_crop, data, request.headers.get('X-Crop-Token'), model_id)
            return JSONResponse(result)
        except CropError as error:
            return JSONResponse({'error': str(error)}, status_code=error.status)
        except (ValueError, ClientDisconnect) as error:
            return JSONResponse({'error': str(error) or '잘못된 요청입니다.'}, status_code=400)
        except Exception as error:
            if local and isinstance(error, TimeoutError):
                return JSONResponse({'error': '사진 전송 시간이 초과되었습니다.'}, status_code=408)
            logger.exception('Image processing failed')
            message = '예측에 실패했습니다. 실행 창의 오류를 확인해주세요.' if local else '이미지 처리에 실패했습니다. 잠시 후 다시 시도해주세요.'
            return JSONResponse({'error': message}, status_code=500)
        finally:
            inference_slot.release()

    if local:
        @app.get('/docs', include_in_schema=False)
        def swagger():
            """로컬 개발 실행에서만 Swagger UI를 제공한다."""
            return get_swagger_ui_html(openapi_url='/openapi.json', title='Face Score API',
                                       swagger_favicon_url='/favicon.ico')

    @app.api_route('/', methods=['GET', 'HEAD'], include_in_schema=False)
    def index():
        """기존 프런트엔드 시작 화면을 제공한다."""
        return FileResponse(FRONTEND / 'index.html')

    @app.api_route('/{filename:path}', methods=['GET', 'HEAD'], include_in_schema=False)
    def assets(filename: str):
        """기존 JavaScript와 CSS만 제공해 외부 경로 접근을 막는다."""
        if filename not in {'js/app.js', 'js/model.js', 'css/styles.css'}:
            raise HTTPException(status_code=404)
        return FileResponse(FRONTEND / filename)
