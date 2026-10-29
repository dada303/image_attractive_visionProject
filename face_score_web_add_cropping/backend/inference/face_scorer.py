"""이미지를 전처리하고 선택된 PyTorch 모델로 기존 점수를 계산한다."""
from io import BytesIO
import math
import sys
from threading import Lock
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms, models
from torch import nn
from ..config import PROJECT, MODEL_IDS, MAX_BYTES, MAX_PIXELS
from ..repositories.model_repository import load_config, load_weights

sys.path.insert(0, str(PROJECT))
from models.DenseNet121.training.model import build_model

class FaceScorer:
    """설정에 맞는 신경망을 구성하고 단일 모델 추론을 수행한다."""
    def __init__(self, checkpoint=None, model_id='densenet121'):
        """설정과 가중치를 읽어 모델, 전처리, 추론 잠금을 준비한다."""
        if model_id not in MODEL_IDS:
            raise ValueError('지원하지 않는 모델입니다.')
        self.config, self.checkpoint = load_config(model_id, checkpoint)
        self.model_id = model_id
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
        state = load_weights(self.checkpoint)
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
        """이미지를 전처리하고 추론한 점수를 기존 응답 필드로 반환한다."""
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
