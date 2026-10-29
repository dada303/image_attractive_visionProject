"""기존 명령행 옵션으로 루프백 전용 로컬 서버를 실행한다."""
import argparse
from pathlib import Path
import uvicorn
from .app import create_app

def main():
    """명령행 옵션으로 모델을 준비하고 로컬 서버를 실행·종료한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--checkpoint', type=Path, default=None)
    args = parser.parse_args()
    print('Loading face score models...', flush=True)
    application = create_app(local=True, port=args.port, checkpoint=args.checkpoint)
    print(f'Face score: http://localhost:{args.port}', flush=True)
    uvicorn.run(application, host='127.0.0.1', port=args.port, workers=1,
                proxy_headers=False, server_header=False, timeout_keep_alive=30)

if __name__ == '__main__':
    main()
