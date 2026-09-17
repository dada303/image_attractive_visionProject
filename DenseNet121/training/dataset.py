# dataset.py
# ------------------------------------------------------------------------------------
# 얼굴 매력도 데이터셋 로딩 모듈
#
# 이 프로젝트에는 4개의 이미지 폴더(images/aihub, images/scut_fbp5500,
# images/남자연예인, images/여자연예인)와, 이 폴더들의 라벨을 1~5점 척도로
# 통일해 정리해 둔 엑셀 파일(labels_combined_1to5.xlsx)이 있다.
#
# 엑셀 파일의 컬럼 구조:
#   dataset      : 이미지가 속한 하위 폴더명 (예: 'aihub', 'scut_fbp5500', ...)
#   filename     : images/{dataset}/ 폴더 안의 실제 파일명
#   score        : 1~5로 정규화된 매력도 점수
#   score_scale  : 원본 척도 표기 (참고용, 학습에는 사용하지 않음)
#
# 이 모듈은 "안 1(aihub만 사용)"과 "안 2(전체 데이터셋 사용)" 두 학습 스크립트가
# 공통으로 사용하는 데이터 로딩/분할/전처리 로직을 제공한다.
# ------------------------------------------------------------------------------------

import os
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet으로 사전 학습된 DenseNet121을 그대로 활용하므로,
# 입력 이미지도 ImageNet 학습 때 사용된 것과 동일한 평균/표준편차로 정규화해야 한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    """학습용/평가용 이미지 전처리 파이프라인을 생성한다.

    - train=True  : 데이터 증강(좌우 반전, 색상 지터, 약간의 회전)을 포함한다.
                     매력도 라벨은 좌우 반전/색상 변화에 크게 영향받지 않으므로 안전한 증강이다.
    - train=False : 검증/테스트/실서비스 추론 시에는 증강 없이 크기 조정 + 정규화만 수행한다.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_labels(labels_path: str, images_root: str, subset: list[str] | None = None) -> pd.DataFrame:
    """라벨 엑셀 파일을 읽어 이미지 절대경로 컬럼을 추가한 DataFrame을 반환한다.

    Args:
        labels_path: labels_combined_1to5.xlsx 경로
        images_root: 이미지들이 들어있는 images/ 폴더 경로
        subset: 학습에 사용할 dataset 값 목록 (예: ['aihub']).
                None이면 전체 데이터셋(4개 폴더 전부)을 사용한다.
                -> 이 subset 인자 하나로 "안 1(aihub만)"과 "안 2(전체)"를 동일한 코드로 처리한다.
    """
    df = pd.read_excel(labels_path)

    if subset is not None:
        df = df[df["dataset"].isin(subset)].reset_index(drop=True)

    df["filepath"] = df.apply(
        lambda row: os.path.join(images_root, str(row["dataset"]), str(row["filename"])),
        axis=1,
    )

    # 엑셀에는 있지만 실제 이미지 파일이 없는 행은 학습 중 오류를 유발하므로 걸러낸다.
    exists_mask = df["filepath"].apply(os.path.isfile)
    missing = int((~exists_mask).sum())
    if missing > 0:
        print(f"[경고] 라벨은 있으나 실제 이미지 파일을 찾지 못해 제외된 항목: {missing}개")
    df = df[exists_mask].reset_index(drop=True)

    return df


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """train / validation / test 세 세트로 분할한다.

    점수(score)가 1~5 정수 등급이므로, 각 세트에 점수 분포가 고르게 섞이도록
    stratify(층화 추출)를 적용한다. 특정 점수의 표본이 너무 적어 층화가
    불가능한 경우(예: 표본 1~2개)에는 일반 무작위 분할로 자동 대체한다.
    """
    train_ratio = 1.0 - val_ratio - test_ratio
    assert train_ratio > 0, "val_ratio + test_ratio 는 1보다 작아야 합니다."

    def safe_split(data, test_size, stratify_col):
        counts = data[stratify_col].value_counts()
        can_stratify = counts.min() >= 2 and len(data) * test_size >= len(counts)
        stratify = data[stratify_col] if can_stratify else None
        return train_test_split(data, test_size=test_size, random_state=seed, stratify=stratify)

    train_df, temp_df = safe_split(df, val_ratio + test_ratio, "score")
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = safe_split(temp_df, relative_test_ratio, "score")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class AttractivenessDataset(Dataset):
    """(이미지, 매력도 점수) 쌍을 반환하는 PyTorch Dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        # score는 회귀(regression) 문제로 다루므로 float 타입으로 반환한다.
        score = float(row["score"])
        return image, score, row["filepath"]


@dataclass
class DataBundle:
    """학습 스크립트에 필요한 로더와 분할 정보를 한 번에 묶어서 전달하기 위한 컨테이너."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def get_dataloaders(
    labels_path: str,
    images_root: str,
    subset: list[str] | None,
    batch_size: int = 32,
    img_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    """라벨 로딩 -> 분할 -> Dataset/DataLoader 생성까지 한 번에 처리하는 헬퍼 함수."""
    df = load_labels(labels_path, images_root, subset=subset)
    if len(df) == 0:
        raise RuntimeError(
            f"선택한 subset={subset} 에 해당하는 이미지를 찾지 못했습니다. "
            "labels_combined_1to5.xlsx 의 dataset 컬럼 값을 확인하세요."
        )

    train_df, val_df, test_df = split_dataframe(df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    train_ds = AttractivenessDataset(train_df, build_transforms(img_size, train=True))
    val_ds = AttractivenessDataset(val_df, build_transforms(img_size, train=False))
    test_ds = AttractivenessDataset(test_df, build_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(train_loader, val_loader, test_loader, train_df, val_df, test_df)
