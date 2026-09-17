# train_all.py
# ------------------------------------------------------------------------------------
# [모델 2 / 안 2] 전체 데이터셋(aihub + scut_fbp5500 + 남자연예인 + 여자연예인, 총 4,858장)을
# 모두 사용해 DenseNet121 얼굴 매력도(1~5점) 회귀 모델을 학습한다.
#
# train_aihub.py(안 1)와 코드 구조는 완전히 동일하며, 유일한 차이는 subset=None으로
# 지정해 라벨 엑셀의 모든 데이터셋을 학습에 사용한다는 점뿐이다.
# 데이터 수가 8배 가까이 많으므로 안 1보다 일반화 성능이 더 좋을 것으로 기대되지만,
# 데이터셋마다 촬영 환경/평가 기준이 달라 라벨 노이즈가 섞일 수 있다는 점은 유의해야 한다.
#
# 실행 예:
#   python train_all.py --epochs 7 --batch-size 32
#
# 결과물은 (DenseNet121 폴더가 아니라) 프로젝트 루트의 outputs/all/ 폴더에 저장되며,
# 파일명에도 'all'이 붙어서
# (best_model_all.pt, training_curve_all.png, test_scatter_all.png, test_metrics_all.json)
# train_aihub.py(안 1)의 결과물과 섞이지 않고 바로 구분할 수 있다.
# ------------------------------------------------------------------------------------

from .train_common import build_arg_parser, run_training

if __name__ == "__main__":
    parser = build_arg_parser(default_run_name="all")
    args = parser.parse_args()

    # subset=None -> load_labels()에서 dataset 컬럼 필터링을 하지 않고 전체를 사용한다.
    run_training(subset=None, run_name="all", args=args)
