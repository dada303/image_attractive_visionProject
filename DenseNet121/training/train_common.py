# train_common.py
# ------------------------------------------------------------------------------------
# train_aihub.py(안 1)와 train_all.py(안 2)가 공통으로 사용하는 학습 루프.
#
# 두 스크립트는 "어떤 이미지들을 학습에 사용할지(subset)"와 "결과를 어디에 저장할지
# (run_name)"만 다르고, 학습/평가/저장 절차는 완전히 동일하다. 그래서 이 로직을
# 한 곳에 모아두고, 각 스크립트는 자신에게 맞는 설정값만 넘겨서 호출하도록 만들었다.
# ------------------------------------------------------------------------------------

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# train_aihub.py/train_all.py를 "python train_aihub.py"처럼 직접 실행할 때도 이 파일과
# dataset.py/engine.py/model.py를 문제없이 import할 수 있도록 DenseNet121/ 폴더를
# sys.path에 넣어둔다 (predict.py, app_gui.py도 동일한 방식을 쓴다).
_DENSENET121_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DENSENET121_DIR not in sys.path:
    sys.path.insert(0, _DENSENET121_DIR)

from training.dataset import get_dataloaders
from training.engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    save_metrics,
    plot_training_curve,
    plot_scatter,
)
from training.model import build_model

# training/train_common.py -> training/ -> DenseNet121/ -> 프로젝트 루트 (3단계 상위)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LABELS_PATH = os.path.join(PROJECT_ROOT, "labels_combined_1to5.xlsx")
DEFAULT_IMAGES_ROOT = os.path.join(PROJECT_ROOT, "images")
# images/, labels_combined_1to5.xlsx 와 마찬가지로 결과물도 DenseNet121 코드 폴더가 아니라
# 프로젝트 루트의 outputs/ 에 저장한다. predict.py, app_gui.py도 동일한 경로를 기본값으로 사용한다.
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")


def set_seed(seed: int):
    """재현 가능한 학습 결과를 위해 난수 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arg_parser(default_run_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DenseNet121 얼굴 매력도(1~5점) 회귀 모델 학습")
    parser.add_argument("--labels-path", default=DEFAULT_LABELS_PATH, help="라벨 엑셀 파일 경로")
    parser.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT, help="images/ 폴더 경로")
    parser.add_argument("--output-dir", default=os.path.join(DEFAULT_OUTPUT_ROOT, default_run_name))
    # CPU 환경에서는 DenseNet121 학습이 오래 걸리므로 기본 epoch 수를 5~7 수준으로 낮게 잡는다.
    # 성능을 더 끌어올리고 싶고 시간 여유가 있다면 --epochs 로 늘려서 실행하면 된다.
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3, help="검증 손실이 이 횟수만큼 개선되지 않으면 조기 종료(early stopping)")
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="DenseNet121의 특징 추출부를 동결(freeze)하고 회귀층만 학습. "
                              "데이터가 적을 때 과적합을 줄이는 데 도움이 되고, 학습 속도도 빨라짐.")
    parser.add_argument("--no-pretrained", action="store_true", help="ImageNet 사전학습 가중치를 사용하지 않음 (기본값: 사용함)")
    return parser


def run_training(subset: list[str] | None, run_name: str, args: argparse.Namespace):
    """
    Args:
        subset: 학습에 사용할 dataset 값 목록. None이면 전체 데이터셋 사용.
        run_name: 결과물(체크포인트, 그래프, 지표)을 식별하기 위한 이름 (예: 'aihub', 'all').
        args: build_arg_parser() 로 만든 CLI 인자.
    """
    set_seed(args.seed)
    # GPU(CUDA)가 있으면 자동으로 사용하고, 없으면 CPU를 사용한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] 사용 디바이스: {device}")

    # 1) 데이터 준비 --------------------------------------------------------------
    bundle = get_dataloaders(
        labels_path=args.labels_path,
        images_root=args.images_root,
        subset=subset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"[{run_name}] 전체 {len(bundle.train_df) + len(bundle.val_df) + len(bundle.test_df)}장 중 "
          f"train={len(bundle.train_df)}, val={len(bundle.val_df)}, test={len(bundle.test_df)}")

    # 2) 모델/옵티마이저 준비 -------------------------------------------------------
    model = build_model(pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    # 검증 손실이 정체되면 학습률을 자동으로 낮춰 미세 조정 효과를 준다.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)
    # 폴더(outputs/aihub, outputs/all)뿐 아니라 파일명 자체에도 run_name을 넣어서,
    # 파일만 따로 복사/공유해도 어떤 데이터셋으로 학습한 모델인지 바로 구분할 수 있게 한다.
    ckpt_path = os.path.join(args.output_dir, f"best_model_{run_name}.pt")

    # 3) 학습 루프 (epoch 단위 + 조기 종료) -----------------------------------------
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improve = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, bundle.train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, bundle.val_loader, criterion, device)
        val_loss = val_metrics["loss_mse"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[{run_name}] epoch {epoch:03d}/{args.epochs} "
              f"train_mse={train_loss:.4f} val_mse={val_loss:.4f} "
              f"val_mae={val_metrics['mae']:.4f} val_acc(±1)={val_metrics['accuracy_within1']:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improve = 0
            save_checkpoint(model, ckpt_path)
            print(f"  -> 검증 손실 개선, 체크포인트 저장: {ckpt_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"[{run_name}] 검증 손실이 {args.patience}epoch 동안 개선되지 않아 조기 종료합니다.")
                break

    elapsed_min = (time.time() - start_time) / 60
    print(f"[{run_name}] 학습 완료 (총 {elapsed_min:.1f}분 소요)")

    # 4) 가장 성능이 좋았던 체크포인트로 테스트셋 최종 평가 ---------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, bundle.test_loader, criterion, device)

    print(f"[{run_name}] === 테스트셋 최종 성능 ===")
    print(f"  MAE            : {test_metrics['mae']:.4f}  (점수 단위, 1~5점 척도)")
    print(f"  RMSE           : {test_metrics['rmse']:.4f}")
    print(f"  Pearson r      : {test_metrics['pearson_r']:.4f}")
    print(f"  정확도(exact)   : {test_metrics['accuracy_exact']*100:.2f}%  (반올림 등급이 정답과 정확히 일치)")
    print(f"  정확도(±1점 허용): {test_metrics['accuracy_within1']*100:.2f}%")

    # 5) 결과물 저장 (지표 JSON, 학습 곡선, 산점도) ---------------------------------
    save_metrics(test_metrics, os.path.join(args.output_dir, f"test_metrics_{run_name}.json"))
    plot_training_curve(history, os.path.join(args.output_dir, f"training_curve_{run_name}.png"))
    plot_scatter(
        test_metrics["y_true"], test_metrics["y_pred"],
        os.path.join(args.output_dir, f"test_scatter_{run_name}.png"),
        title=f"DenseNet121 ({run_name}) - Predicted vs Actual",
    )
    print(f"[{run_name}] 결과물 저장 위치: {args.output_dir}")

    return test_metrics
