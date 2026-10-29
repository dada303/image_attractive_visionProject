"""모델 가용 상태를 관리하고 Crop 확인 후 선택한 모델로 예측한다."""
from ..config import MODEL_IDS
from ..inference.face_scorer import FaceScorer
from .crop_service import CropService

class ModelRegistry:
    """사용 가능한 모델과 로딩 실패를 관리하고 예측 흐름을 연결한다."""
    def __init__(self, checkpoint=None, model_ids=MODEL_IDS):
        """Crop 서비스와 활성 모델을 준비하고 모델별 로딩 실패를 기록한다."""
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
        """활성 모델의 표시 이름과 가용 상태를 반환한다."""
        labels = {'densenet121': 'DenseNet121', 'mobilenetv3': 'MobileNetV3', 'efficientnet_b0': 'EfficientNet-B0'}
        return [{'id': key, 'label': labels[key], 'ready': key in self.scorers,
                 'error': '모델 파일 또는 설정을 확인해주세요.' if key in self.errors else None}
                for key in self.model_ids]

    def predict(self, data, model_id='densenet121'):
        """요청 모델의 지원·가용 여부를 확인하고 추론을 호출한다."""
        if model_id not in MODEL_IDS:
            raise ValueError('지원하지 않는 모델입니다.')
        if model_id not in self.scorers:
            raise ValueError('선택한 모델을 사용할 수 없습니다. 모델 파일과 설정을 확인해주세요.')
        return self.scorers[model_id].predict(data)

    def predict_crop(self, data, token, model_id='densenet121'):
        """전달된 Crop의 확인 토큰을 검증한 뒤 재검출 없이 예측한다."""
        self.crops.verify(data, token)
        return self.predict(data, model_id)
