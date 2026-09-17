# model.py
# ------------------------------------------------------------------------------------
# DenseNet121 기반 얼굴 매력도 회귀 모델 정의
#
# 매력도(1~5점)는 "3점 vs 4점"처럼 등급 사이에 순서와 거리 개념이 있는 값이므로,
# 5개 클래스로 분류(classification)하기보다는 하나의 실수 값을 예측하는
# 회귀(regression) 문제로 다룬다. 이렇게 하면 "3.5점처럼 애매한 얼굴"도
# 자연스럽게 표현할 수 있고, 예측값을 반올림하면 1~5 등급으로도 바로 변환할 수 있다.
#
# torchvision이 제공하는 ImageNet 사전학습 DenseNet121의 마지막 분류층(classifier)만
# 출력 1개짜리 선형층으로 교체하여 전이학습(transfer learning)에 사용한다.
# ------------------------------------------------------------------------------------

import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """DenseNet121 회귀 모델을 생성한다.

    Args:
        pretrained: True면 ImageNet으로 학습된 가중치를 불러와 전이학습에 사용한다.
                    (처음 실행 시 인터넷에서 가중치를 자동 다운로드한다.)
        freeze_backbone: True면 DenseNet121의 특징 추출부(features)를 동결하여
                    마지막 회귀층만 학습한다. 데이터 수가 적은 aihub 단독 학습(안 1)처럼
                    과적합(overfitting)이 우려될 때 사용하면 도움이 된다.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # 원래 DenseNet121의 classifier는 (in_features -> 1000, ImageNet 클래스 수) 선형층이다.
    # 이를 (in_features -> 1) 로 교체해 "매력도 점수" 하나만 출력하도록 만든다.
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),   # 작은 데이터셋에서의 과적합을 줄이기 위한 드롭아웃
        nn.Linear(in_features, 1),
    )
    return model
