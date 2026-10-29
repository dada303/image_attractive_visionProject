"""기존 모듈 실행 경로와 create_app 진입점을 유지하는 운영 서버 실행부."""
import logging
import os
import uvicorn
from .app import create_app

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    application = create_app()
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8767'))
    print(f'Production origin: http://{host}:{port}', flush=True)
    uvicorn.run(application, host=host, port=port, workers=1,
                proxy_headers=False, server_header=False, timeout_keep_alive=30)
