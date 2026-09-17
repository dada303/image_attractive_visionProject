"""Production entry point: Waitress -> Flask -> existing ModelRegistry."""
import logging
import os
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import unquote, urlsplit

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from local_model import MAX_BYTES, MODEL_IDS, ModelRegistry

FRONTEND = Path(__file__).resolve().parent / 'frontend'


def create_app(registry=None):
    port = int(os.environ.get('PORT', '8767'))
    public_origin = os.environ.get('PUBLIC_ORIGIN', '').rstrip('/')
    if public_origin:
        parsed = urlsplit(public_origin)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('PUBLIC_ORIGIN must be an HTTPS origin, e.g. https://face.example.com')
    origins = {f'http://localhost:{port}', f'http://127.0.0.1:{port}'}
    if public_origin:
        origins.add(public_origin)
    hosts = {urlsplit(origin).netloc.lower() for origin in origins}
    default_model = os.environ.get('DEFAULT_MODEL', 'efficientnet_b0')
    enabled = tuple(x.strip() for x in os.environ.get('ENABLED_MODELS', ','.join(MODEL_IDS)).split(',') if x.strip())
    if not enabled or any(x not in MODEL_IDS for x in enabled) or default_model not in enabled:
        raise ValueError('Check ENABLED_MODELS and DEFAULT_MODEL')
    registry = registry if registry is not None else ModelRegistry(model_ids=enabled)
    if default_model not in registry.scorers:
        raise RuntimeError('Default model failed to load. Check checkpoint/config before serving.')
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES
    app.extensions['models'] = registry
    inference_slot = BoundedSemaphore(1)

    @app.before_request
    def validate_request():
        print("REQUEST HOST =", repr(request.host.lower()))
        print("ALLOWED HOSTS =", repr(hosts))
        if request.host.lower() not in hosts:
            return jsonify(error='허용되지 않은 접속 주소입니다. PUBLIC_ORIGIN 설정을 확인해주세요.'), 403
        origin = request.headers.get('Origin')
        if origin and origin not in origins:
            return jsonify(error='허용되지 않은 요청 출처입니다.'), 403

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {413: '10MB 이하의 사진을 선택해주세요.', 404: '없는 경로입니다.', 400: '잘못된 요청입니다.'}
        return jsonify(error=messages.get(error.code, '요청을 처리할 수 없습니다.')), error.code

    @app.get('/api/health')
    @app.get('/api/models')
    def health():
        return jsonify(ready=True, default_model=default_model, models=registry.list_models())

    @app.post('/api/predict')
    def predict():
        model_id = request.args.get('model', default_model)
        if model_id not in registry.scorers:
            return jsonify(error='사용할 수 없는 모델입니다.'), 400
        filename = unquote(request.headers.get('X-Filename', ''))
        if Path(filename).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            return jsonify(error='JPG, PNG, WebP 확장자의 사진을 선택해주세요.'), 400
        if request.mimetype != 'application/octet-stream':
            return jsonify(error='이미지 바이너리 요청이 필요합니다.'), 415
        if not request.content_length:
            return jsonify(error='빈 이미지입니다.'), 400
        if request.content_length > MAX_BYTES:
            return jsonify(error='10MB 이하의 사진을 선택해주세요.'), 413
        if not inference_slot.acquire(blocking=False):
            return jsonify(error='다른 사진을 분석 중입니다. 잠시 후 다시 시도해주세요.'), 429
        try:
            # Decoder verifies actual bytes; filename is never used as a disk path.
            result = registry.predict(request.get_data(cache=False), model_id)
            return jsonify(result)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception('Inference failed')
            return jsonify(error='예측에 실패했습니다. 잠시 후 다시 시도해주세요.'), 500
        finally:
            inference_slot.release()

    @app.get('/')
    def index():
        return send_from_directory(FRONTEND, 'index.html')

    @app.get('/<path:filename>')
    def assets(filename):
        if filename not in {'js/app.js', 'js/model.js', 'css/styles.css'}:
            return jsonify(error='없는 경로입니다.'), 404
        return send_from_directory(FRONTEND, filename)

    return app


if __name__ == '__main__':
    from waitress import serve
    logging.basicConfig(level=logging.INFO)
    application = create_app()  # Load each enabled checkpoint once before accepting requests.
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8767'))
    print(f'Production origin: http://{host}:{port}', flush=True)
    serve(application, host=host, port=port, threads=4, connection_limit=32,
          channel_timeout=30, max_request_body_size=MAX_BYTES,
          inbuf_overflow=MAX_BYTES + 1, ident='face-score')
