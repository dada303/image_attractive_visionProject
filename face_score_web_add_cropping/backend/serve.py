"""Loopback-only web UI and PyTorch inference service."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import traceback
from .face_crop import CropError, validate_upload
from .local_model import DEFAULT_CHECKPOINT, MAX_BYTES, ModelRegistry

FRONTEND = Path(__file__).resolve().parents[1] / 'frontend'

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, scorer, **kwargs):
        self.scorer = scorer
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def local_request(self):
        port = self.server.server_port
        hosts = {f'localhost:{port}', f'127.0.0.1:{port}'}
        origin = self.headers.get('Origin')
        if self.headers.get('Host') not in hosts or (origin and origin not in {f'http://{h}' for h in hosts}):
            self.respond(403, {'error': 'localhost에서 접속해주세요.'})
            return False
        return True

    def respond(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.local_request():
            return
        if urlsplit(self.path).path in ('/api/health', '/api/models'):
            self.respond(200, {'ready': bool(self.scorer.scorers), 'models': self.scorer.list_models()})
        else:
            super().do_GET()

    def do_POST(self):
        if not self.local_request():
            return
        if urlsplit(self.path).path not in ('/api/crop', '/api/predict'):
            self.respond(404, {'error': '없는 API입니다.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.respond(400, {'error': '잘못된 요청 크기입니다.'})
            return
        if not 0 < length <= MAX_BYTES:
            self.respond(413, {'error': '10MB 이하의 사진을 선택해주세요.'})
            return
        try:
            validate_upload(self.headers.get('X-Filename'), self.headers.get('Content-Type'), length)
        except CropError as error:
            self.respond(error.status, {'error': str(error)})
            return
        self.connection.settimeout(30)
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError('사진 전송이 완료되지 않았습니다.')
            if urlsplit(self.path).path == '/api/crop':
                png, token, metadata = self.scorer.crops.prepare(data)
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(png)))
                self.send_header('X-Crop-Token', token)
                self.send_header('X-Face-Count', str(metadata['face_count']))
                self.end_headers()
                self.wfile.write(png)
            else:
                self.respond(200, self.scorer.predict_crop(data, self.headers.get('X-Crop-Token'), parse_qs(urlsplit(self.path).query).get('model', ['densenet121'])[0]))
        except CropError as exc:
            self.respond(exc.status, {'error': str(exc)})
        except ValueError as exc:
            self.respond(400, {'error': str(exc)})
        except TimeoutError:
            self.respond(408, {'error': '사진 전송 시간이 초과되었습니다.'})
        except Exception:
            traceback.print_exc()
            self.respond(500, {'error': '예측에 실패했습니다. 실행 창의 오류를 확인해주세요.'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--checkpoint', type=Path, default=None)
    args = parser.parse_args()
    print('Loading face score models...', flush=True)
    scorer = ModelRegistry(args.checkpoint)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, scorer=scorer))
    print(f'Face score: http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
