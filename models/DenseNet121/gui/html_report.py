# html_report.py
# ------------------------------------------------------------------------------------
# 여러 장을 한 번에 측정한 결과를 카드 형태의 그리드로 정리한 "단일 HTML 파일"로 저장하기
# 위한 모듈. 사진을 <img> 태그 안에 base64로 직접 삽입하기 때문에, HTML 파일 하나만
# 다른 폴더나 다른 사람에게 전달해도 이미지가 깨지지 않고 그대로 열린다.
# ------------------------------------------------------------------------------------

import base64
import html
import io
import os
import sys
from datetime import datetime

from PIL import Image

# app_gui.py를 "python app_gui.py"로 직접 실행하는 경우와 "python -m DenseNet121...."로
# 실행하는 경우 모두에서 inference/ 패키지를 문제없이 import할 수 있도록, 이 파일이 속한
# DenseNet121/ 폴더를 sys.path에 넣어둔다 (predict.py, app_gui.py도 동일한 방식을 쓴다).
_DENSENET121_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DENSENET121_DIR not in sys.path:
    sys.path.insert(0, _DENSENET121_DIR)

from inference.predict import MODEL_LABELS

THUMBNAIL_MAX_SIZE = 260


def _pil_to_data_uri(image: Image.Image) -> str:
    """PIL 이미지를 HTML <img> 태그에 바로 넣을 수 있는 base64 data URI 문자열로 바꾼다."""
    thumb = image.copy()
    thumb.thumbnail((THUMBNAIL_MAX_SIZE, THUMBNAIL_MAX_SIZE))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_card(image_path: str, item: dict, run_names: list[str]) -> str:
    """측정한 사진 한 장에 대한 카드(사진 + 예측/실제 점수 + 데이터셋/파일명) HTML을 만든다."""
    filename_text = html.escape(item["filename"] or os.path.basename(image_path))

    if not item["models"]:
        # 모든 모델에서 측정이 실패한 사진(체크포인트 없음, 손상된 파일 등)은 크래시 대신
        # 실패했다는 사실을 카드로 보여준다. 원본 사진을 열 수 없을 수도 있으므로 썸네일도 생략한다.
        return f'''
    <div class="card">
      <div class="score-label" style="color:#c0392b;">측정 실패</div>
      <div class="meta">{filename_text}</div>
    </div>'''

    # 대표 이미지가 아니었던 사진들은 Grad-CAM을 계산하지 않았으므로, 원본 얼굴 이미지를 사용한다.
    any_result = next(iter(item["models"].values()))
    thumb_src = _pil_to_data_uri(any_result["face_image"])

    score_rows = []
    for run_name in run_names:
        result = item["models"].get(run_name)
        if result is None:
            continue
        label = html.escape(MODEL_LABELS.get(run_name, run_name))
        # 모델이 예측한 소수점 점수와, 그것을 반올림한 정수 등급을 함께 보여준다.
        score_rows.append(
            f'<div class="score-label">{label} 예측 점수</div>'
            f'<div class="score-value">{result["score"]:.4f} / {result["rounded"]}</div>'
        )

    dataset_text = html.escape(item["dataset"]) if item["dataset"] else "업로드 사진"

    return f'''
    <div class="card">
      <img src="{thumb_src}" alt="{filename_text}">
      {"".join(score_rows)}
      <div class="meta">{dataset_text}<br>{filename_text}</div>
    </div>'''


def _find_reference_metrics(batch_results: dict, run_name: str) -> dict | None:
    """해당 모델(run_name)의 테스트셋 성능 요약을 batch_results 어디서든 하나 찾아 반환한다.

    reference_metrics는 이미지마다 다시 계산하는 값이 아니라 train_*.py가 학습을 마치고
    저장해 둔 고정된 지표(모든 이미지에 대해 동일한 값)이므로, 배치 안에서 한 번만 찾으면 된다.
    """
    for item in batch_results.values():
        result = item["models"].get(run_name)
        if result is not None and result["reference_metrics"] is not None:
            return result["reference_metrics"]
    return None


def _build_model_summary(batch_results: dict, run_names: list[str]) -> str:
    """HTML 제목 바로 아래에 넣을, 이번에 사용한 모델들의 테스트셋 성능 요약 블록을 만든다."""
    rows = []
    for run_name in run_names:
        label = html.escape(MODEL_LABELS.get(run_name, run_name))
        ref = _find_reference_metrics(batch_results, run_name)
        if ref is not None:
            rows.append(
                f'<div class="summary-row"><strong>{label}</strong> — '
                f'테스트셋 기준 MAE {ref["mae"]:.3f} · RMSE {ref["rmse"]:.3f} · '
                f'정확도(정확히 일치) {ref["accuracy_exact"]*100:.1f}% · '
                f'정확도(±1점 허용) {ref["accuracy_within1"]*100:.1f}%</div>'
            )
        else:
            rows.append(f'<div class="summary-row"><strong>{label}</strong> — 성능 정보 없음</div>')
    return "\n".join(rows)


def build_html_report(batch_results: dict, run_names: list[str]) -> str:
    """여러 장의 측정 결과를 카드 그리드로 배치한 self-contained HTML 문자열을 만든다.

    Args:
        batch_results: {image_path: {"dataset":.., "filename":.., "actual_score":..,
                        "models": {run_name: predict_one()의 반환값 dict}}}
        run_names: 이번 측정에 사용한 모델 이름 목록 (예: ["aihub"], ["aihub", "all"])
    """
    cards = "\n".join(_build_card(path, item, run_names) for path, item in batch_results.items())
    model_summary = _build_model_summary(batch_results, run_names)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>얼굴 매력도 측정 결과</title>
<style>
  body {{ font-family: "Malgun Gothic", "Segoe UI", sans-serif; background:#f2f3f5; margin:0; padding:28px; }}
  h1 {{ text-align:center; color:#1a1a1a; margin-bottom:4px; }}
  .subtitle {{ text-align:center; color:#777; margin-bottom:20px; font-size:14px; }}
  .model-summary {{ max-width:700px; margin:0 auto 24px auto; background:#fff; border:1px solid #e2e2e2;
                     border-radius:10px; padding:14px 20px; }}
  .summary-row {{ color:#333; font-size:14px; padding:4px 0; }}
  /* 화면 폭과 무관하게 항상 가로 3장씩 배치하고, 그 아래로 다음 3장씩 계속 나열한다. */
  .grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:18px; max-width:1100px; margin:0 auto; }}
  .card {{ background:#fff; border:1px solid #e2e2e2; border-radius:10px; padding:14px; text-align:center;
           box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  .card img {{ width:100%; border-radius:8px; margin-bottom:10px; object-fit:cover; aspect-ratio:1/1; }}
  .score-label {{ color:#2a7ae2; font-size:13px; margin-top:8px; }}
  .score-value {{ font-size:19px; font-weight:bold; color:#111; margin-bottom:2px; }}
  .meta {{ color:#8a8a8a; font-size:12px; margin-top:10px; line-height:1.5; }}
</style>
</head>
<body>
  <h1>얼굴 매력도 측정 결과</h1>
  <div class="model-summary">
    {model_summary}
  </div>
  <div class="subtitle">생성 시각: {timestamp} · 총 {len(batch_results)}장</div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>'''
