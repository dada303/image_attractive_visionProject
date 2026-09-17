# train_aihub.py
# ------------------------------------------------------------------------------------
# [모델 1 / 안 1] AIHub 데이터셋 사진만 사용해 DenseNet121 얼굴 매력도(1~5점) 회귀 모델을 학습한다.
#
# images/aihub 폴더의 606장(라벨 존재 기준)만 학습에 사용하며, 다른 데이터셋
# (scut_fbp5500, 남자연예인, 여자연예인)은 사용하지 않는다.
# 데이터 수가 적은 편이라 과적합 위험이 크므로, 필요하다면
#   python train_aihub.py --freeze-backbone
# 옵션으로 DenseNet121의 특징 추출부를 동결하고 마지막 층만 학습시키는 것도 고려해볼 수 있다.
#
# 실행 예:
#   python train_aihub.py --epochs 7 --batch-size 32
#
# 결과물은 (DenseNet121 폴더가 아니라) 프로젝트 루트의 outputs/aihub/ 폴더에 저장되며,
# 파일명에도 'aihub'가 붙어서
# train_all.py(안 2)의 결과물과 섞이지 않고 바로 구분할 수 있다.
#   - best_model_aihub.pt       : 검증 성능이 가장 좋았던 시점의 모델 가중치
#   - training_curve_aihub.png  : epoch별 train/val loss 그래프
#   - test_scatter_aihub.png    : 테스트셋 (실제 점수 vs 예측 점수) 산점도
#   - test_metrics_aihub.json   : MAE, RMSE, 상관계수, 정확도 등 최종 성능 지표
# ------------------------------------------------------------------------------------

import os
import sys

_DENSENET121_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DENSENET121_DIR not in sys.path:
    sys.path.insert(0, _DENSENET121_DIR)

from training.train_common import build_arg_parser, run_training

if __name__ == "__main__":
    parser = build_arg_parser(default_run_name="aihub")
    args = parser.parse_args()

    # 이 스크립트의 핵심: dataset 컬럼이 'aihub'인 이미지들만 학습에 사용한다.
    run_training(subset=["aihub"], run_name="aihub", args=args)
