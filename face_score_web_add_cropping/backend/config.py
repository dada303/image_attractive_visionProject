"""파일 이동과 무관한 기준 경로와 기존 처리 제한값을 정의한다."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FRONTEND = Path(__file__).resolve().parents[1] / 'frontend'
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_IDS = ('densenet121', 'mobilenetv3', 'efficientnet_b0')
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
