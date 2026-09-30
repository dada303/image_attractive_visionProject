# -*- coding: utf-8 -*-
"""Vision 1/idol_images의 사진에서 가장 큰 얼굴 하나를 크롭한다."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_ascii_cascade_path() -> str:
    """한글 Windows 사용자 경로를 피한 Haar Cascade 경로를 반환한다."""
    source = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.name != "nt":
        return source

    public_dir = os.environ.get("PUBLIC", r"C:\Users\Public")
    destination_dir = Path(public_dir) / "cv2_cascades"
    destination = destination_dir / "haarcascade_frontalface_default.xml"
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        return str(destination)
    except OSError as error:
        print(f"[경고] Cascade 복사 실패: {error}")
        return source


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로에서도 이미지를 읽는다."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def write_jpg(path: Path, image: np.ndarray) -> bool:
    """한글 경로에도 JPEG를 저장한다."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    try:
        buffer.tofile(path)
        return True
    except OSError:
        return False


def largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    """여러 얼굴 중 면적이 가장 큰 얼굴을 선택한다."""
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(width), int(height)


def square_box(
    face: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """얼굴 박스에 여백을 더하고 이미지 안의 정사각형 좌표로 변환한다."""
    x, y, width, height = face
    side = round(max(width, height) * (1 + 2 * margin))
    side = max(1, min(side, image_width, image_height))
    center_x, center_y = x + width / 2, y + height / 2
    left = max(0, min(round(center_x - side / 2), image_width - side))
    top = max(0, min(round(center_y - side / 2), image_height - side))
    return left, top, left + side, top + side


def crop_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    size: int = 224,
    margin: float = 0.25,
    overwrite: bool = False,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if size <= 0:
        raise ValueError("size는 1 이상이어야 합니다.")
    if margin < 0:
        raise ValueError("margin은 0 이상이어야 합니다.")

    cascade_path = get_ascii_cascade_path()
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Haar Cascade를 불러오지 못했습니다: {cascade_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    saved = skipped = not_found = failed = 0

    for index, source in enumerate(images, start=1):
        destination = output_dir / f"{source.stem}.jpg"
        if destination.exists() and not overwrite:
            print(f"[{index}/{len(images)}] 기존 파일 건너뜀: {source.name}")
            skipped += 1
            continue

        image = read_image(source)
        if image is None:
            print(f"[{index}/{len(images)}] 이미지 읽기 실패: {source.name}")
            failed += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40),
        )
        face = largest_face(faces)
        if face is None:
            print(f"[{index}/{len(images)}] 얼굴 미검출: {source.name}")
            not_found += 1
            continue

        image_height, image_width = image.shape[:2]
        left, top, right, bottom = square_box(
            face, image_width, image_height, margin
        )
        crop = image[top:bottom, left:right]
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

        if write_jpg(destination, crop):
            print(f"[{index}/{len(images)}] 저장: {destination.name}")
            saved += 1
        else:
            print(f"[{index}/{len(images)}] 저장 실패: {destination.name}")
            failed += 1

    print(
        f"\n완료: 입력 {len(images)}, 저장 {saved}, 기존 파일 {skipped}, "
        f"얼굴 미검출 {not_found}, 실패 {failed}"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base_dir / "idol_images")
    parser.add_argument(
        "--output", type=Path, default=base_dir / "cropped_faces_haar"
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    crop_folder(
        args.input,
        args.output,
        size=args.size,
        margin=args.margin,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
