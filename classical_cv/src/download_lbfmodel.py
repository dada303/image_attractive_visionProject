"""OpenCV 얼굴검출/랜드마크 모델 파일을 다운로드한다.

- haarcascade_frontalface_default.xml (얼굴 검출, Viola-Jones)
- lbfmodel.yaml (68점 랜드마크, ~54MB)

용량이 커서(특히 lbfmodel.yaml) git에는 커밋하지 않고(.gitignore 처리) 각자 이 스크립트로 받는다.
반드시 combined/ 디렉터리에서 실행할 것 (`python classical_cv/src/download_lbfmodel.py`).
"""
import os
import urllib.request

MODEL_DIR = "classical_cv/models"

FILES = {
    "haarcascade_frontalface_default.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    ),
    "lbfmodel.yaml": "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.isfile(path) and os.path.getsize(path) > 10_000:
            print("이미 존재합니다:", path)
            continue
        print("다운로드 중:", url)
        urllib.request.urlretrieve(url, path)
        print("완료:", path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
