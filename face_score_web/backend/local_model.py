"""Local DenseNet121 inference. Uploaded images stay in memory."""
from io import BytesIO
import math
import json
from pathlib import Path
import sys
from threading import Lock
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms, models
from torch import nn
from .face_crop import CropService

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from DenseNet121.training.model import build_model

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
DEFAULT_CHECKPOINT = MODELS_DIR / 'densenet121' / 'best_model_all.pt'
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

class FaceScorer:
    def __init__(self, checkpoint=None, model_id='densenet121'):
        if model_id not in MODEL_IDS:
            raise ValueError('지원하지 않는 모델입니다.')
        folder = MODELS_DIR / model_id
        self.config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        self.model_id = model_id
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else (folder / self.config['checkpoint']).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {self.checkpoint}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        arch = self.config['architecture']
        if arch == 'densenet121':
            self.model = build_model(pretrained=False)
        elif arch == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(weights=None)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 1)
        elif arch == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        else:
            raise ValueError(f'지원하지 않는 architecture: {arch}')
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint_data = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        state = checkpoint_data
        for key in ('state_dict', 'model_state_dict', 'model_state'):
            if key in checkpoint_data:
                state = checkpoint_data[key]
                break
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        cfg = self.config
        interpolation = transforms.InterpolationMode(cfg['interpolation'])
        if cfg['resize'] == 'center_crop':
            steps = [transforms.Resize(cfg['resize_size'], interpolation=interpolation),
                     transforms.CenterCrop(cfg['image_size'])]
        elif cfg['resize'] == 'stretch':
            steps = [transforms.Resize((cfg['image_size'], cfg['image_size']), interpolation=interpolation)]
        else:
            raise ValueError('지원하지 않는 resize 설정입니다.')
        if cfg['output'] not in ('identity', 'sigmoid_1_5'):
            raise ValueError('지원하지 않는 output 설정입니다.')
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])])
        self.lock = Lock()

    def predict(self, data):
        if not data or len(data) > MAX_BYTES:
            raise ValueError('10MB 이하의 사진을 선택해주세요.')
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('JPG, PNG, WebP 사진만 사용할 수 있습니다.')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('2,000만 화소 이하의 사진을 선택해주세요.')
                image = ImageOps.exif_transpose(image).convert('RGB')
                tensor = self.transform(image).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('사진을 읽을 수 없습니다. 정상적인 이미지 파일을 선택해주세요.') from exc
        with self.lock, torch.inference_mode():
            output = self.model(tensor)
            raw_score = output.item()
            score = (1 + 4 * output.sigmoid()).item() if self.config['output'] == 'sigmoid_1_5' else raw_score
        if not math.isfinite(raw_score):
            raise RuntimeError('모델이 유효한 점수를 반환하지 않았습니다.')
        return {'score': min(5.0, max(1.0, score)), 'raw_score': raw_score,
                'model': self.config['label'], 'model_id': self.model_id, 'checkpoint': self.checkpoint.name,
                'device': str(self.device), 'score_scale': [1, 5]}


MODEL_IDS = ('densenet121', 'mobilenetv3', 'efficientnet_b0')

class ModelRegistry:
    def __init__(self, checkpoint=None, model_ids=MODEL_IDS):
        self.crops = CropService()
        self.model_ids = tuple(dict.fromkeys(model_ids))
        self.scorers = {}
        self.errors = {}
        for model_id in self.model_ids:
            try:
                self.scorers[model_id] = FaceScorer(checkpoint if model_id == 'densenet121' else None, model_id)
            except Exception as exc:
                self.errors[model_id] = str(exc)
                print(f'Model unavailable [{model_id}]: {exc}', flush=True)

    def list_models(self):
        labels = {'densenet121': 'DenseNet121', 'mobilenetv3': 'MobileNetV3', 'efficientnet_b0': 'EfficientNet-B0'}
        return [{'id': key, 'label': labels[key], 'ready': key in self.scorers,
                 'error': '모델 파일 또는 설정을 확인해주세요.' if key in self.errors else None}
                for key in self.model_ids]

    def predict(self, data, model_id='densenet121'):
        if model_id not in MODEL_IDS:
            raise ValueError('지원하지 않는 모델입니다.')
        if model_id not in self.scorers:
            raise ValueError('선택한 모델을 사용할 수 없습니다. 모델 파일과 설정을 확인해주세요.')
        return self.scorers[model_id].predict(data)

    def predict_crop(self, data, token, model_id='densenet121'):
        self.crops.verify(data, token)
        return self.predict(data, model_id)
