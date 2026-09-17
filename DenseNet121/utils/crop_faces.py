# crop_faces.py
# ------------------------------------------------------------------------------------
# 지정한 폴더 안의 사진들에서 얼굴만 검출해 잘라낸 뒤, 원본은 그대로 두고
# 같은 폴더 아래의 cropped/ 하위 폴더에 크롭된 사진만 따로 저장한다.
#
# 실행 예:
#   python crop_faces.py --input-dir "C:\Users\...\Downloads\5 point K-MALE"
# ------------------------------------------------------------------------------------

import argparse
import os
import sys

_DENSENET121_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DENSENET121_DIR not in sys.path:
    sys.path.insert(0, _DENSENET121_DIR)

from inference.predict import detect_and_crop_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def crop_folder(input_dir: str, output_dir: str, margin: float = 0.3) -> None:
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not filenames:
        print(f"[안내] 이미지 파일을 찾지 못했습니다: {input_dir}")
        return

    found_count = 0
    for idx, filename in enumerate(filenames, start=1):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            face_image, face_found = detect_and_crop_face(src_path, margin=margin)
        except Exception as e:
            print(f"  [{idx}/{len(filenames)}] [오류] {filename}: {e}")
            continue

        face_image.save(dst_path)
        if face_found:
            found_count += 1
            print(f"  [{idx}/{len(filenames)}] 얼굴 검출 후 저장: {filename}")
        else:
            print(f"  [{idx}/{len(filenames)}] [경고] 얼굴 미검출, 원본 전체 저장: {filename}")

    print(f"\n총 {len(filenames)}장 중 {found_count}장 얼굴 검출 성공. 저장 위치: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="폴더 안의 사진들에서 얼굴을 검출해 크롭 사진만 따로 저장한다.")
    parser.add_argument("--input-dir", required=True, help="원본 사진들이 들어있는 폴더 경로")
    parser.add_argument("--output-dir", default=None,
                         help="크롭 사진을 저장할 폴더 경로 (기본값: <input-dir>/cropped)")
    parser.add_argument("--margin", type=float, default=0.3,
                         help="검출된 얼굴 박스 주변에 추가로 포함할 여백 비율 (기본값: 0.3)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "cropped")
    crop_folder(args.input_dir, output_dir, margin=args.margin)


if __name__ == "__main__":
    main()
