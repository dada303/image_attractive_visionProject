# engine.py
# ------------------------------------------------------------------------------------
# 학습 1 epoch 실행, 성능 평가(정확도/오차 지표 계산), 결과 시각화, 체크포인트 저장 등
# 두 학습 스크립트(train_aihub.py, train_all.py)가 공통으로 사용하는 로직 모음.
#
# [성능 지표 설명]
#   - MAE  (Mean Absolute Error)      : 예측 점수와 실제 점수 차이의 평균 절댓값. 작을수록 좋음.
#   - RMSE (Root Mean Squared Error)  : 오차 제곱의 평균에 루트를 씌운 값. 큰 오차에 더 민감함.
#   - Pearson r                       : 예측 점수와 실제 점수 사이의 선형 상관관계 (1에 가까울수록 좋음).
#   - Accuracy(exact)                 : 예측값을 반올림해 1~5 등급으로 만들었을 때, 실제 등급과
#                                        정확히 일치하는 비율. 가장 엄격한 "정확도".
#   - Accuracy(±1)                    : 반올림한 예측 등급이 실제 등급과 1점 이내 차이인 비율.
#                                        사람이 매기는 매력도 평가도 평가자마다 1점 정도 오차가
#                                        흔하므로, 실질적인 성능을 보기에 더 적합한 지표.
# ------------------------------------------------------------------------------------

import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """학습 데이터 1 epoch를 순회하며 가중치를 업데이트하고 평균 손실을 반환한다."""
    model.train()
    total_loss = 0.0
    for images, scores, _ in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        scores = scores.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """검증/테스트셋에 대해 손실과 다양한 성능 지표를 계산한다."""
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    for images, scores, _ in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        scores = scores.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, scores)
        total_loss += loss.item() * images.size(0)

        all_true.extend(scores.cpu().numpy().tolist())
        all_pred.extend(outputs.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # 표본이 2개 미만이거나 값이 전부 동일하면 상관계수를 계산할 수 없으므로 예외 처리.
    try:
        corr = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        corr = float("nan")

    # 회귀 출력을 1~5 정수 등급으로 반올림하여 "정확도" 개념의 지표도 함께 계산한다.
    y_pred_rounded = np.clip(np.round(y_pred), 1, 5)
    y_true_rounded = np.clip(np.round(y_true), 1, 5)
    acc_exact = float(np.mean(y_pred_rounded == y_true_rounded))
    acc_within1 = float(np.mean(np.abs(y_pred_rounded - y_true_rounded) <= 1))

    return {
        "loss_mse": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": corr,
        "accuracy_exact": acc_exact,
        "accuracy_within1": acc_within1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_metrics(metrics: dict, path: str):
    """혼동을 줄이기 위해 원본 예측/정답 배열은 별도 키로 두고, 요약 지표만 보기 좋게 저장한다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "y_true": metrics["y_true"], "y_pred": metrics["y_pred"]},
                   f, ensure_ascii=False, indent=2)


def plot_training_curve(history: dict, save_path: str):
    """epoch별 train/val loss 변화를 그래프로 저장한다 (과적합 여부 확인용)."""
    import matplotlib
    matplotlib.use("Agg")  # 화면 없는 서버 환경에서도 저장 가능하도록 비대화형 백엔드 사용
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss (MSE)")
    plt.plot(epochs, history["val_loss"], label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scatter(y_true, y_pred, save_path: str, title: str):
    """테스트셋의 (실제 점수, 예측 점수) 산점도를 그려 모델 성능을 시각적으로 확인한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=15)
    plt.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title(title)
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
