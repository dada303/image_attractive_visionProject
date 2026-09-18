# %% [markdown]
# # EfficientNet-B0 얼굴 점수 학습 (Colab)
# 런타임에서 GPU를 선택하고 위에서 아래로 실행하세요. Google Drive에 프로젝트의 images 폴더와 엑셀을 올려둡니다.
# 원본 파일은 수정하지 않으며 결과는 Drive의 colab_outputs 아래 실행별 폴더에 저장합니다.
# 
# - DATA_MODE: all_domains / aihub_only / scut_fbp5500_only
# - TASK: regression(연속 점수, 기본) / classification(정수 1~5만 허용)
# - 이미지가 얼굴 중심으로 준비되어 있다는 전제입니다. 자동 얼굴 검출은 포함하지 않습니다.
# - 점수는 제공된 라벨의 평가 경향에 대한 예측값입니다.
# - 실제 엑셀은 생성 환경에서 열어보지 못했으므로 검증 셀의 결과를 먼저 확인하세요.
# 
# %%
# %pip -q install "openpyxl>=3.1,<4" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "Pillow>=10,<13" "tqdm>=4.66,<5"
# torch / torchvision은 Colab에 설치된 짝을 유지합니다.
import torch, torchvision
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
assert torch.cuda.is_available(), "GPU 런타임으로 변경한 뒤 다시 실행하세요."

# %% [markdown]
# ## 라벨·이미지 검증
# 전체 도메인을 먼저 검사하고 공통 분할을 만든 뒤 학습 모드를 적용합니다.
# 중복 파일명은 임의 선택하지 않습니다. filename을 images 기준 상대 경로로 수정하면 해결됩니다.
# score_scale은 값 분포를 표시합니다. 실제 학습 타깃은 score이며 1~5 범위를 검사합니다.
# 
# %%
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import json, random, hashlib, sys, subprocess, uuid
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

PROJECT = Path("/content/drive/MyDrive/image_attractive_visionProject")
LABELS = PROJECT / "labels_combined_1to5.xlsx"  # .csv도 지원
IMAGES = PROJECT / "images"  # 실제 이름이 image라면 수정
SHEET = 0
DATA_MODE = "all_domains"   # aihub_only / scut_fbp5500_only / all_domains
TASK = "regression"         # 또는 classification
SEED = 42
EPOCHS = 15
BATCH_SIZE = 32             # GPU 메모리 부족 시 16 또는 8
LR = 1e-4
PATIENCE = 4
NUM_WORKERS = 2
FULL_IMAGE_CHECK = False  # True: 모든 이미지 디코딩·픽셀 중복 검사
# person_id가 있으면 도메인 간에도 같은 인물은 같은 ID여야 합니다.
# 없으면 동일 픽셀 이미지 단위로만 분리하며, 동일 인물의 다른 사진 누수는 막지 못합니다.
GROUP_COLUMN = "person_id"

TRAINING_RECIPE = "staged"  # baseline / staged
REGRESSION_LOSS = "mse"  # mse / huber
HUBER_DELTA = 0.5
HEAD_ONLY_EPOCHS = 3 if TRAINING_RECIPE == "staged" else 0
BACKBONE_LR = 1e-5 if TRAINING_RECIPE == "staged" else LR
HEAD_LR = 1e-4 if TRAINING_RECIPE == "staged" else LR
FREEZE_BN_STATS = TRAINING_RECIPE == "staged"
GRAD_CLIP = 1.0 if TRAINING_RECIPE == "staged" else None
USE_SCHEDULER = TRAINING_RECIPE == "staged"
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
MILD_AUGMENTATION = False
if TRAINING_RECIPE == "staged":
    EPOCHS, PATIENCE = 30, 7
assert TRAINING_RECIPE in {"baseline", "staged"}
assert REGRESSION_LOSS in {"mse", "huber"}
assert 0 <= HEAD_ONLY_EPOCHS < EPOCHS and min(BACKBONE_LR, HEAD_LR, HUBER_DELTA) > 0

OUT = PROJECT / "colab_outputs" / (
    datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
)
assert DATA_MODE in {"aihub_only", "scut_fbp5500_only", "all_domains"}
assert TASK in {"regression", "classification"}
assert LABELS.is_file(), f"엑셀 없음: {LABELS}"
assert IMAGES.is_dir(), f"이미지 폴더 없음: {IMAGES}"
OUT.mkdir(parents=True, exist_ok=False)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
config = dict(project=str(PROJECT), labels=str(LABELS), images=str(IMAGES),
              sheet=SHEET, data_mode=DATA_MODE, task=TASK, seed=SEED, epochs=EPOCHS,
              batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, group_column=GROUP_COLUMN,
              model="efficientnet_b0", weights="IMAGENET1K_V1",
              torch=torch.__version__, torchvision=torchvision.__version__)
config.update({
    "full_image_check": FULL_IMAGE_CHECK, "num_workers": NUM_WORKERS, "optimizer": "AdamW", "weight_decay": 1e-4,
    "loss": "CrossEntropyLoss" if TASK == "classification" else "MSELoss",
    "output": "5-class expected score" if TASK == "classification" else "1 + 4 * sigmoid",
    "score_scale": [1, 5], "fine_tuning": "all backbone and head parameters",
    "preprocessing": "EXIF transpose, RGB, resize 256 bicubic, center crop 224",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "augmentation": "training only: horizontal flip p=0.5",
    "split": "group split approximately 70/15/15; path or person_id grouped; pixel duplicates only when full_image_check=True",
    "selection_metric": "validation MAE", "device": str(torch.cuda.get_device_name(0)),
})

config.update({
    "training_recipe": TRAINING_RECIPE, "head_only_epochs": HEAD_ONLY_EPOCHS,
    "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "lr": HEAD_LR,
    "freeze_bn_stats": FREEZE_BN_STATS, "grad_clip": GRAD_CLIP,
    "fine_tuning": "head then backbone" if HEAD_ONLY_EPOCHS else "all parameters",
    "scheduler": "ReduceLROnPlateau" if USE_SCHEDULER else None,
    "scheduler_patience": SCHEDULER_PATIENCE, "scheduler_factor": SCHEDULER_FACTOR,
    "loss": "CrossEntropyLoss" if TASK == "classification" else REGRESSION_LOSS,
    "huber_delta": HUBER_DELTA, "mild_augmentation": MILD_AUGMENTATION,
    "augmentation": "horizontal flip + affine 5 degrees/3% translation" if MILD_AUGMENTATION else "horizontal flip",
})
(OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
(OUT / "environment.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
print("결과 폴더:", OUT)

# %%
import unicodedata

def read_label_table(path, sheet=0):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return pd.read_excel(path, sheet_name=sheet)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        print("CSV 인코딩: CP949")
        return pd.read_csv(path, encoding="cp949")

df = read_label_table(LABELS, SHEET)
df.columns = df.columns.astype(str).str.strip()
assert not df.columns.duplicated().any(), "중복 열 이름이 있습니다."
required = {"dataset", "filename", "score", "score_scale"}
assert required <= set(df.columns), f"필수 열 누락: {required - set(df.columns)}"
assert df["dataset"].notna().all() and df["filename"].notna().all(), "출처/파일명 누락"
df["dataset"] = df["dataset"].astype(str).map(lambda v: unicodedata.normalize("NFC", v).strip().casefold())
df["filename"] = df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)
assert df["filename"].ne("").all(), "빈 파일명"
allowed = {"aihub", "scut_fbp5500", "남자연예인", "여자연예인", "5_female", "5_male"}
assert set(df.dataset) <= allowed, f"예상하지 못한 도메인: {set(df.dataset) - allowed}"
df["score"] = pd.to_numeric(df["score"], errors="coerce")
assert df.score.notna().all() and df.score.between(1, 5).all(), "score 누락 또는 1~5 범위 위반"
print("score_scale 분포:")
display(df["score_scale"].value_counts(dropna=False))
assert df.score_scale.notna().all(), "score_scale 누락"
if TASK == "classification":
    assert np.allclose(df.score, np.round(df.score), rtol=0, atol=1e-6), (
        "소수 점수가 있습니다. regression을 선택하세요. 임의 반올림하지 않습니다."
    )

# CSV/Excel labels and domain-aware, Unicode-normalized image matching.
import unicodedata
from pathlib import PurePosixPath

def name_key(value):
    return unicodedata.normalize("NFC", str(value).strip().replace("\\", "/")).casefold()

def build_image_index(images):
    root = images.resolve()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted(p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    relative, domain_names, flat_names = {}, {}, {}
    for p in files:
        if not p.resolve().is_relative_to(root):
            raise ValueError(f"images 외부 파일 링크: {p}")
        rel = p.relative_to(images)
        relative.setdefault(name_key(rel.as_posix()), []).append(p)
        if len(rel.parts) > 1:
            domain_names.setdefault((name_key(rel.parts[0]), name_key(p.name)), []).append(p)
        else:
            flat_names.setdefault(name_key(p.name), []).append(p)
    return files, relative, domain_names, flat_names

def resolve_indexed_image(dataset, filename, index):
    _, relative, domain_names, flat_names = index
    domain, name = name_key(dataset), name_key(filename)
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts or ":" in name:
        raise ValueError(f"허용되지 않는 이미지 경로: {filename}")
    candidates = list(relative.get(domain + "/" + name, []))
    # Explicit images-relative path must agree with dataset.
    if len(parts) > 1:
        if parts[0] == domain:
            candidates.extend(relative.get(name, []))
    else:
        candidates.extend(domain_names.get((domain, name), []))
        # Support the original flat images layout, but never another domain.
        if not candidates:
            candidates.extend(flat_names.get(name, []))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(f"파일 누락/정규화 이름 중복: dataset={dataset}, filename={filename}, 후보 수={len(candidates)}")
    return candidates[0]

image_index = build_image_index(IMAGES)
files = image_index[0]
paths, problems = [], []
for i, row in enumerate(df.itertuples(index=False)):
    try:
        paths.append(str(resolve_indexed_image(row.dataset, row.filename, image_index)))
    except ValueError as exc:
        paths.append(None)
        problems.append({"row": i + 2, "dataset": row.dataset,
                         "filename": row.filename, "error": str(exc)})
errors = pd.DataFrame(problems, columns=["row", "dataset", "filename", "error"])
errors.to_csv(OUT / "validation_errors.csv", index=False, encoding="utf-8-sig")
if problems:
    # Actual names and Unicode code points make Drive-only failures diagnosable.
    inventory = [{"relative_path": p.relative_to(IMAGES).as_posix(),
                  "normalized_path": name_key(p.relative_to(IMAGES).as_posix()),
                  "filename_codepoints": " ".join(f"U+{ord(c):04X}" for c in p.name)} for p in files]
    pd.DataFrame(inventory, columns=["relative_path", "normalized_path", "filename_codepoints"]).to_csv(
        OUT / "image_inventory.csv", index=False, encoding="utf-8-sig")
    display(errors.head(20))
    print("Drive 실제 파일명: image_inventory.csv 확인")
assert not problems, f"파일 연결 오류 {len(problems)}건: validation_errors.csv 및 image_inventory.csv 확인"
df["path"] = paths
assert not df.path.duplicated().any(), "동일 파일을 가리키는 중복 라벨 행을 정리하세요."
# False is also the fallback when only this cell is pasted into an older session.
full_image_check = globals().get("FULL_IMAGE_CHECK", False)
config["full_image_check"] = full_image_check
config["image_grouping"] = "pixel_hash" if full_image_check else "resolved_path"
(OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
if full_image_check:
    hashes, bad = {}, []
    for path in tqdm(sorted(set(paths)), desc="이미지 디코딩 및 중복 검사"):
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                hashes[path] = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        except Exception as exc:
            bad.append({"path": path, "error": str(exc)})
    pd.DataFrame(bad).to_csv(OUT / "corrupt_images.csv", index=False)
    assert not bad, "손상 이미지가 있습니다. corrupt_images.csv 확인"
    df["image_group_key"] = df.path.map(hashes)
    conflicts = df.groupby("image_group_key").score.nunique()
    assert not (conflicts > 1).any(), "동일 이미지에 상충하는 score가 있습니다."
else:
    # Only metadata is read here; pixels are opened later by the training Dataset.
    df["image_group_key"] = df["path"].map(lambda p: "path:" + str(Path(p).resolve()))
    print("빠른 검증: 라벨·파일 매칭 완료. 전체 디코딩 및 픽셀 중복 검사는 생략합니다.")
    print("손상 이미지는 학습 중 발견될 수 있으며, 다른 파일명의 동일 사진은 중복 판별하지 않습니다.")
print("라벨 수:", len(df), "이미지 파일 수:", len(files))
display(df.groupby("dataset").score.agg(["count", "min", "mean", "max"]))
df.to_csv(OUT / "validated_manifest.csv", index=False, encoding="utf-8-sig")

matched_paths = {Path(p).resolve() for p in paths}
unlabeled = [p.relative_to(IMAGES).as_posix() for p in files if p.resolve() not in matched_paths]
print(f"파일 매칭 완료: {len(paths)}/{len(df)}행, 라벨 없는 이미지: {len(unlabeled)}장")
if unlabeled:
    pd.DataFrame({"relative_path": unlabeled}).to_csv(
        OUT / "unlabeled_images.csv", index=False, encoding="utf-8-sig")

# %%
from sklearn.model_selection import GroupShuffleSplit
# 경로(전체 검사 시 픽셀 해시)와 person_id를 연결한 그룹을 만듭니다.
# 두 조건 중 하나라도 같으면 동일 분할에 들어갑니다.
df = df.reset_index(drop=True)
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
def union(a, b):
    parent[find(a)] = find(b)
keys = {}
has_people = GROUP_COLUMN in df.columns
for i, row in df.iterrows():
    row_keys = [("image", row.image_group_key)]
    if has_people and pd.notna(row[GROUP_COLUMN]) and str(row[GROUP_COLUMN]).strip():
        row_keys.append(("person", str(row[GROUP_COLUMN]).strip()))
    for key in row_keys:
        if key in keys:
            union(i, keys[key])
        else:
            keys[key] = i
df["group"] = [find(i) for i in range(len(df))]
if not has_people or df[GROUP_COLUMN].isna().any() or df[GROUP_COLUMN].astype(str).str.strip().eq("").any():
    print("주의: 일부/전체 person_id 없음. 동일 인물의 다른 사진 누수 가능성이 있어 성능은 예비 결과입니다.")
assert df.group.nunique() >= 10, "그룹 수가 너무 적습니다. 데이터를 늘리거나 분할 설계를 조정하세요."
train_idx, rest_idx = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=SEED)
                           .split(df, groups=df.group))
rest = df.iloc[rest_idx]
val_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=SEED)
                         .split(rest, groups=rest.group))
df["split"] = ""
df.loc[train_idx, "split"] = "train"
df.loc[rest.index[val_idx], "split"] = "val"
df.loc[rest.index[test_idx], "split"] = "test"
assert df.groupby("group").split.nunique().max() == 1
assert df.groupby("image_group_key").split.nunique().max() == 1
df.to_csv(OUT / "all_domain_splits.csv", index=False)
mode_domains = {
    "aihub_only": "aihub",
    "scut_fbp5500_only": "scut_fbp5500",
}
selected = (df if DATA_MODE == "all_domains"
            else df[df.dataset == mode_domains[DATA_MODE]])
parts = {s: selected[selected.split == s].copy() for s in ["train", "val", "test"]}
assert all(len(v) for v in parts.values()), "선택 모드에 빈 분할이 있습니다."
for s, part in parts.items():
    part.to_csv(OUT / f"{s}.csv", index=False)
display(pd.crosstab(selected.dataset, selected.split))
# 그룹 크기에 따라 실제 비율은 70/15/15와 달라질 수 있습니다.
# 두 모드 비교 시 원본 행 순서/데이터/SEED/GROUP_COLUMN을 동일하게 유지하세요.

# %%
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
eval_transform = weights.transforms()
augmentations = [transforms.RandomHorizontalFlip()]
if MILD_AUGMENTATION:
    augmentations.append(transforms.RandomAffine(degrees=5, translate=(0.03, 0.03),
        interpolation=transforms.InterpolationMode.BILINEAR))
train_transform = transforms.Compose(augmentations + [eval_transform])
class FaceDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as im:
            image = self.transform(ImageOps.exif_transpose(im).convert("RGB"))
        target = (torch.tensor(int(round(row.score)) - 1, dtype=torch.long)
                  if TASK == "classification" else torch.tensor(float(row.score), dtype=torch.float32))
        return image, target

def make_model(pretrained):
    model = efficientnet_b0(weights=weights if pretrained else None)
    width = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(width, 5 if TASK == "classification" else 1)
    return model
def score_from_output(output):
    if TASK == "classification":
        return (output.softmax(1) * torch.arange(1, 6, device=output.device)).sum(1)
    return 1 + 4 * output.squeeze(1).sigmoid()
device = torch.device("cuda")
model = make_model(True).to(device)
generator = torch.Generator().manual_seed(SEED)
loaders = {
    s: DataLoader(FaceDataset(part, train_transform if s == "train" else eval_transform),
                  batch_size=BATCH_SIZE, shuffle=s == "train", num_workers=NUM_WORKERS,
                  pin_memory=True, generator=generator if s == "train" else None)
    for s, part in parts.items()
}

criterion = (nn.CrossEntropyLoss() if TASK == "classification" else
             nn.HuberLoss(delta=HUBER_DELTA) if REGRESSION_LOSS == "huber" else nn.MSELoss())
optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": BACKBONE_LR},
    {"params": model.classifier.parameters(), "lr": HEAD_LR},
], weight_decay=1e-4)
scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
    threshold=1e-4, threshold_mode="abs", min_lr=1e-7) if USE_SCHEDULER else None)

def set_training_stage(model, epoch):
    head_only = epoch <= HEAD_ONLY_EPOCHS
    for parameter in model.features.parameters():
        parameter.requires_grad_(not head_only)
    model.train()
    if head_only:
        model.features.eval()
    elif FREEZE_BN_STATS:
        for layer in model.features.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
    return "head_only" if head_only else "fine_tune"

@torch.no_grad()
def evaluate(loader):
    model.eval()
    truth, predictions, classes = [], [], []
    for images, target in loader:
        output = model(images.to(device))
        predictions.extend(score_from_output(output).cpu().tolist())
        truth.extend((target + 1 if TASK == "classification" else target).tolist())
        if TASK == "classification":
            classes.extend((output.argmax(1) + 1).cpu().tolist())
    return np.array(truth), np.array(predictions), np.array(classes)
# 한 배치로 입력/출력/손실 형태를 검증합니다.
images, targets = next(iter(loaders["train"]))
model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
    loss = criterion(outputs if TASK == "classification" else score_from_output(outputs), targets.to(device))
assert torch.isfinite(loss)
assert score_from_output(outputs).ge(1).all() and score_from_output(outputs).le(5).all()
print("배치 검증 통과:", tuple(images.shape), tuple(outputs.shape), "loss:", loss.item())

# %%

# Small CPU smoke test: run before training; does not change model weights or random state.
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(7)
    class TinyStageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
            self.classifier = nn.Linear(4, 1)
        def forward(self, x):
            return self.classifier(self.features(x))
    probe = TinyStageModel()
    x = torch.randn(8, 4)
    stage = set_training_stage(probe, 1)
    initial_mean = probe.features[1].running_mean.clone()
    probe(x).sum().backward()
    if HEAD_ONLY_EPOCHS:
        assert all(p.grad is None for p in probe.features.parameters())
        assert torch.equal(initial_mean, probe.features[1].running_mean)
    assert probe.classifier.weight.grad is not None
    probe.zero_grad(set_to_none=True)
    set_training_stage(probe, HEAD_ONLY_EPOCHS + 1)
    initial_mean = probe.features[1].running_mean.clone()
    probe(x).sum().backward()
    assert probe.features[0].weight.grad is not None
    if FREEZE_BN_STATS:
        assert torch.equal(initial_mean, probe.features[1].running_mean)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4)
    probe_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        probe_optimizer, patience=0, factor=0.5)
    probe_scheduler.step(1.0)
    probe_scheduler.step(1.0)
    assert abs(probe_optimizer.param_groups[0]["lr"] - 5e-5) < 1e-12
print("PASS: stage gradients, BatchNorm handling, learning-rate reduction")

# %%

history, best, stale = [], float("inf"), 0
for epoch in range(1, EPOCHS + 1):
    stage = set_training_stage(model, epoch)
    if epoch == HEAD_ONLY_EPOCHS + 1:
        stale = 0
    running, count = 0., 0
    for images, targets in tqdm(loaders["train"], desc=f"Epoch {epoch}/{EPOCHS} ({stage})"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs if TASK == "classification" else score_from_output(outputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        running += loss.item() * len(images)
        count += len(images)
    truth, predicted, _ = evaluate(loaders["val"])
    mae = float(np.abs(truth - predicted).mean())
    if not np.isfinite(mae):
        raise RuntimeError("Non-finite validation MAE")
    row = {"epoch": epoch, "stage": stage, "train_loss": running / count, "val_mae": mae,
           "val_rmse": float(np.sqrt(np.mean((truth - predicted) ** 2))),
           "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"]}
    history.append(row)
    pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
    print(row)
    if mae < best:
        best, stale = mae, 0
        checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "config": config, "epoch": epoch, "val_mae": best}
        torch.save(checkpoint, OUT / "best.pt")
        validation_predictions = parts["val"].reset_index(drop=True).copy()
        validation_predictions["prediction"] = predicted
        validation_predictions.to_csv(OUT / "best_validation_predictions.csv", index=False)
    elif stage == "fine_tune":
        stale += 1
    if scheduler is not None and stage == "fine_tune":
        scheduler.step(mae)
    if stage == "fine_tune" and stale >= PATIENCE:
        print("Early stopping")
        break
print("Best validation MAE:", best)
# Match this fingerprint before comparing different recipes. Test results are not used for selection.
split_signature = hashlib.sha256(
    selected[["dataset", "filename", "score", "split"]].sort_values(["dataset", "filename"])
    .to_csv(index=False).encode("utf-8")).hexdigest()
run_summary = {"recipe": TRAINING_RECIPE, "task": TASK, "data_mode": DATA_MODE,
               "seed": SEED, "split_signature": split_signature, "best_val_mae": best,
               "best_epoch": min(history, key=lambda r: r["val_mae"])["epoch"],
               "epochs_completed": len(history)}
(OUT / "validation_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
print(json.dumps(run_summary, indent=2))

# %%
# 검증으로 선정한 모델을 고정하고 테스트는 마지막에 평가합니다.
checkpoint = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
truth, predicted, classes = evaluate(loaders["test"])
result = parts["test"].reset_index(drop=True).copy()
result["prediction"] = predicted
result["absolute_error"] = np.abs(truth - predicted)
if TASK == "classification":
    result["predicted_class"] = classes
result.to_csv(OUT / "test_predictions.csv", index=False)
def metrics(frame):
    actual = frame["score"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    # Pearson correlation is undefined for fewer than two samples or constant values.
    pearson = (float(np.corrcoef(actual, predicted)[0, 1])
               if len(actual) >= 2 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
               else None)
    values = {"n": len(frame), "mse": mse, "mae": float(np.mean(np.abs(errors))),
              "rmse": float(np.sqrt(mse)), "pearson_r": pearson}
    if TASK == "classification":
        values["accuracy"] = float((frame.predicted_class == frame.score).mean())
    return values

def format_test_metrics(name, values):
    r = "N/A" if values["pearson_r"] is None else f"{values['pearson_r']:.4f}"
    return (f"[{name}] [mode={checkpoint['config']['data_mode']}] "
            f"[task={checkpoint['config']['task']}] [model={checkpoint['config']['model']}] "
            f"Test n={values['n']}  MSE={values['mse']:.4f}  MAE={values['mae']:.4f}  "
            f"RMSE={values['rmse']:.4f}  Pearson r={r}")
# Report the configuration saved with the evaluated checkpoint.
run_config = dict(checkpoint["config"])
run_config.update({
    "evaluated_checkpoint": str(OUT / "best.pt"),
    "best_epoch": checkpoint["epoch"],
    "best_validation_mae": checkpoint["val_mae"],
    "split_counts": {s: part.groupby("dataset").size().to_dict() for s, part in parts.items()},
    "training_datasets": sorted(parts["train"]["dataset"].unique().tolist()),
    "test_datasets": sorted(result["dataset"].unique().tolist()),
})
report_text = json.dumps(run_config, ensure_ascii=False, indent=2)
print("=== Evaluated run configuration ===")
print(report_text)
(OUT / "evaluation_config.json").write_text(report_text, encoding="utf-8")
summary = {"overall": metrics(result),
           "domains": {name: metrics(part) for name, part in result.groupby("dataset")}}
(OUT / "test_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
if DATA_MODE == "all_domains":
    print(format_test_metrics("all_domains", summary["overall"]))
for domain, values in summary["domains"].items():
    print(format_test_metrics(domain, values))
pd.DataFrame(history).plot(x="epoch", y=["train_loss", "val_mae"], subplots=True, grid=True)
print("저장 위치:", OUT)

# %%
# Test images with original labels and predictions, without rerunning inference.
import base64
import html
import io
import math
from IPython.display import HTML, display

def prediction_card(row):
    with Image.open(row.path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((240, 240))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    domain = html.escape(str(row.dataset))
    filename = html.escape(str(row.filename))
    return (f'<article class="prediction-card"><img alt="{filename}" src="data:image/jpeg;base64,{encoded}">'
            f'<div class="score-label">AI 모델 예측 점수 / 실제 점수</div>'
            f'<div class="score-values">{row.prediction:.4f} / {row.score:.4f}</div>'
            f'<div class="image-caption">{domain}<br>{filename}</div></article>')

GALLERY_STYLE = """<style>
.prediction-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;}
.prediction-card {border:1px solid #bbb;border-radius:8px;padding:12px;background:#fff;color:#111;}
.prediction-card img {width:100%;height:220px;object-fit:contain;}
.prediction-card {min-width:0;text-align:center;}
.score-label {margin-top:8px;font-size:12px;overflow-wrap:anywhere;}
.score-values {margin-top:4px;font-size:18px;font-weight:700;}
.image-caption {margin-top:10px;font-size:12px;color:#555;overflow-wrap:anywhere;}
</style>"""

def gallery_html(frame, title):
    cards = "".join(prediction_card(row) for row in frame.itertuples(index=False))
    context = html.escape(json.dumps(run_config, ensure_ascii=False, indent=2))
    return ('<!doctype html><html lang="ko"><meta charset="utf-8">' + GALLERY_STYLE
            + '<h2>' + html.escape(title) + '</h2>'
            + '<details><summary>학습 및 평가 설정</summary><pre>' + context + '</pre></details>'
            + '<p>원본 이미지 미리보기 · 점수 범위 1~5 · 실제 점수와 모델 예측 비교</p>'
            + '<div class="prediction-grid">' + cards + '</div></html>')

def show_predictions(domain=None, page=1, page_size=24):
    frame = result if domain is None else result[result.dataset == domain]
    if frame.empty:
        raise ValueError("해당 도메인의 테스트 이미지가 없습니다.")
    if page_size < 1 or page < 1 or page > math.ceil(len(frame) / page_size):
        raise ValueError("유효한 page / page_size를 지정하세요.")
    subset = frame.iloc[(page - 1) * page_size:page * page_size]
    title = f"{domain or 'all test datasets'} · {page}/{math.ceil(len(frame) / page_size)} 페이지"
    display(HTML(gallery_html(subset, title)))

# All test images are exported, while notebook output is paginated.
gallery_path = OUT / "test_image_gallery.html"
gallery_path.write_text(gallery_html(result, "전체 테스트 이미지 점수"), encoding="utf-8")
# Persist to Drive OUT and also to the Colab runtime output folder.
local_output = Path("/content/output")
local_output.mkdir(parents=True, exist_ok=True)
local_gallery_path = local_output / f"{OUT.name}_test_image_gallery.html"
local_gallery_path.write_bytes(gallery_path.read_bytes())
print("Drive 결과 파일:", gallery_path)
print("Colab output 결과 파일:", local_gallery_path)
from IPython.display import FileLink
display(FileLink(str(local_gallery_path)))
show_predictions(page=1)
# 다음 페이지: show_predictions(page=2)
# 도메인 선택: show_predictions(domain="aihub", page=1)

# %% [markdown]
# ## 이미지 한 장 예측
# 학습한 최적 체크포인트와 동일한 전처리를 사용합니다. 분류 모드는 5개 확률의 기댓값을 score로 반환합니다.
# 런타임 재접속 후에는 설정/모델 정의 셀을 실행하고 CHECKPOINT를 이전 best.pt로 지정하세요.
# 본인이 생성한 체크포인트만 로드하세요.
# 
# %%
CHECKPOINT = OUT / "best.pt"  # 이전 실행 결과의 절대 경로로 바꿀 수 있습니다.
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
TASK = checkpoint["config"]["task"]
model = make_model(False).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
@torch.no_grad()
def predict_image(path):
    with Image.open(path) as im:
        tensor = eval_transform(ImageOps.exif_transpose(im).convert("RGB")).unsqueeze(0).to(device)
    output = model(tensor)
    response = {"score": float(score_from_output(output).item()),
                "score_scale": [1, 5], "model": "efficientnet_b0",
                "data_mode": checkpoint["config"]["data_mode"], "task": TASK}
    if TASK == "classification":
        response["class"] = int(output.argmax(1).item()) + 1
        response["probabilities"] = output.softmax(1).cpu().tolist()[0]
    return response
SAMPLE_IMAGE = parts["test"].iloc[0].path  # 원하는 이미지 경로로 변경
print(predict_image(SAMPLE_IMAGE))
