"""로컬·운영 설정을 확인하고 공통 FastAPI 컨트롤러와 서비스를 조립한다."""
import os
from urllib.parse import urlsplit
from fastapi import FastAPI
from .config import MODEL_IDS
from .services.model_service import ModelRegistry
from .controllers.image_controller import register_routes

def create_app(registry=None, *, local=False, port=None, checkpoint=None):
    """기존 모드별 기본값을 유지하고 로컬에서만 API 문서를 제공한다."""
    port = port if port is not None else (8766 if local else int(os.environ.get('PORT', '8767')))
    public_origin = '' if local else os.environ.get('PUBLIC_ORIGIN', '').rstrip('/')
    if public_origin:
        parsed = urlsplit(public_origin)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('PUBLIC_ORIGIN must be an HTTPS origin, e.g. https://face.example.com')
    origins = {f'http://localhost:{port}', f'http://127.0.0.1:{port}'}
    if public_origin:
        origins.add(public_origin)
    hosts = {urlsplit(origin).netloc.lower() for origin in origins}
    default_model = 'densenet121' if local else os.environ.get('DEFAULT_MODEL', 'efficientnet_b0')
    enabled = MODEL_IDS if local else tuple(x.strip() for x in os.environ.get('ENABLED_MODELS', ','.join(MODEL_IDS)).split(',') if x.strip())
    if not enabled or any(x not in MODEL_IDS for x in enabled) or default_model not in enabled:
        raise ValueError('Check ENABLED_MODELS and DEFAULT_MODEL')
    registry = registry if registry is not None else (ModelRegistry(checkpoint) if local else ModelRegistry(model_ids=enabled))
    if not local and default_model not in registry.scorers:
        raise RuntimeError('Default model failed to load. Check checkpoint/config before serving.')
    app = FastAPI(title='Face Score API', docs_url=None, redoc_url=None,
                  openapi_url='/openapi.json' if local else None, redirect_slashes=False)
    app.state.models = registry
    register_routes(app, registry, hosts, origins, default_model, local=local)
    return app
