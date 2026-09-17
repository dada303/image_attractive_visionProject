"""매니페스트(csv)의 모든 이미지에 대해 얼굴검출->정렬->특징추출을 수행하고
결과를 .npz로 저장한다.

사용법 (반드시 combined/ 에서 실행):
    python classical_cv/src/extract_features.py data/manifest_aihub.csv results/features_aihub.npz
    python classical_cv/src/extract_features.py data/manifest_all.csv   results/features_all.npz

SCUT-FBP5500 이미지는 자체 검출(Haar+LBF) 대신, 데이터셋이 공식 제공하는
사람이 직접 찍은 86점 랜드마크(../SCUT-FBP5500_v2/facial landmark/*.pts)를 사용한다.
그 외 소스(aihub, 연예인)는 기존과 동일하게 Haar+LBF로 검출/정렬한다.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_align import FaceAligner, align_from_eyes  # noqa: E402
from features import extract_all_features, gabor_features  # noqa: E402
from scut_landmarks import load_pts86, geometric_features_86, eye_centers_86  # noqa: E402

SCUT_LANDMARK_DIR = os.path.join("..", "SCUT-FBP5500_v2", "facial landmark")


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def process_scut_row(img, filename):
    """SCUT 공식 86점 랜드마크 기반 처리. 실패하면 None, None (호출측에서 검출 방식으로 폴백)."""
    stem = os.path.splitext(filename)[0]
    pts_path = os.path.join(SCUT_LANDMARK_DIR, stem + ".pts")
    if not os.path.isfile(pts_path):
        return None, None

    pts86 = load_pts86(pts_path)
    geo = geometric_features_86(pts86)

    left_eye, right_eye = eye_centers_86(pts86)
    aligned, _ = align_from_eyes(img, left_eye, right_eye)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    gab = gabor_features(gray)
    return geo, gab


def run(manifest_csv, out_npz):
    df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
    aligner = FaceAligner()

    geo_list, gab_list, score_list = [], [], []
    fail_rows = []
    n_scut_landmark = 0

    t0 = time.time()
    for i, row in df.iterrows():
        img = load_image(row["path"])
        if img is None:
            fail_rows.append((row["dataset"], row["filename"], "read_fail"))
            continue

        geo = gab = None
        if row["dataset"] == "scut_fbp5500":
            geo, gab = process_scut_row(img, row["filename"])
            if geo is not None:
                n_scut_landmark += 1

        if geo is None:
            aligned, pts = aligner.detect_and_align(img)
            if aligned is None:
                fail_rows.append((row["dataset"], row["filename"], "detect_fail"))
                continue
            geo, gab = extract_all_features(aligned, pts)

        geo_list.append(geo)
        gab_list.append(gab)
        score_list.append(row["score"])

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(df)} 처리 완료 ({elapsed:.1f}s, 실패 {len(fail_rows)}건)")

    geo_arr = np.vstack(geo_list)
    gab_arr = np.vstack(gab_list)
    scores = np.array(score_list, dtype=np.float64)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        geometric=geo_arr,
        gabor=gab_arr,
        scores=scores,
    )

    total = len(df)
    n_fail = len(fail_rows)
    print(f"완료: 총 {total}장 중 성공 {total - n_fail}, 실패 {n_fail} ({n_fail/total:.1%})")
    if n_scut_landmark:
        print(f"  (이 중 SCUT {n_scut_landmark}장은 자체 검출 대신 공식 86점 랜드마크 사용)")
    print(f"저장: {out_npz}")

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows, columns=["dataset", "filename", "reason"])
        fail_csv = out_npz.replace(".npz", "_failures.csv")
        fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print(f"실패 목록 저장: {fail_csv}")
        print(fail_df["dataset"].value_counts())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
