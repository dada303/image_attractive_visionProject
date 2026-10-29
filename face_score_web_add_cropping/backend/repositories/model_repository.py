"""모델 설정 JSON과 체크포인트를 기존 방식 그대로 읽는다."""
import json
from pathlib import Path
import torch
from ..config import MODELS_DIR


def load_config(model_id, checkpoint=None):
    """모델 설정과 체크포인트 경로를 읽고 파일 존재 여부를 확인한다."""
    folder = MODELS_DIR / model_id
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else (folder / config['checkpoint']).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {checkpoint_path}')
    return config, checkpoint_path


def load_weights(checkpoint):
    """안전 로딩 설정으로 체크포인트를 읽어 지원하는 가중치 형식을 추출한다."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint_data
    for key in ('state_dict', 'model_state_dict', 'model_state'):
        if key in checkpoint_data:
            state = checkpoint_data[key]
            break
    return state
