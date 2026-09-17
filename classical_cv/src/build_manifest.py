"""라벨 엑셀에서 이미지 경로가 포함된 매니페스트(CSV)를 생성한다.

두 시나리오용 매니페스트를 만든다:
  - manifest_aihub.csv : aihub 소스만 (pose/표정 변화가 큰 어려운 세트)
  - manifest_all.csv   : aihub + scut_fbp5500 + 남자연예인 + 여자연예인 전체
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # combined/
LABELS_XLSX = os.path.join(BASE_DIR, "labels_combined_1to5.xlsx")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def build():
    df = pd.read_excel(LABELS_XLSX, sheet_name="전체")
    df = df[["dataset", "filename", "score"]].copy()
    df["path"] = df.apply(lambda r: os.path.join(IMAGES_DIR, r["dataset"], r["filename"]), axis=1)

    missing = ~df["path"].apply(os.path.isfile)
    if missing.any():
        print(f"[warn] {missing.sum()}개 파일을 찾을 수 없어 제외합니다.")
        df = df[~missing]

    os.makedirs(OUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUT_DIR, "manifest_all.csv"), index=False, encoding="utf-8-sig")
    print(f"manifest_all.csv: {len(df)} rows")

    aihub = df[df["dataset"] == "aihub"]
    aihub.to_csv(os.path.join(OUT_DIR, "manifest_aihub.csv"), index=False, encoding="utf-8-sig")
    print(f"manifest_aihub.csv: {len(aihub)} rows")


if __name__ == "__main__":
    build()
