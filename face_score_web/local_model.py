"""Local DenseNet121 inference. Uploaded images stay in memory."""
from io import BytesIO
import math
from pathlib import Path
import sys
from threading import Lock
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from DenseNet121.training.model import build_model

DEFAULT_CHECKPOINT = PROJECT / 'DenseNet121' / 'outputs' / 'best_model_all.pt'
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

class FaceScorer:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {self.checkpoint}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = build_model(pretrained=False)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Same preprocessing as training.dataset.build_transforms(train=False).
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
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
            raw_score = self.model(tensor).item()
        if not math.isfinite(raw_score):
            raise RuntimeError('모델이 유효한 점수를 반환하지 않았습니다.')
        return {'score': min(5.0, max(1.0, raw_score)), 'raw_score': raw_score,
                'model': 'DenseNet121', 'checkpoint': self.checkpoint.name,
                'device': str(self.device), 'score_scale': [1, 5]}
