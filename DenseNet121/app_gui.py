# app_gui.py
# ------------------------------------------------------------------------------------
# 얼굴 매력도 예측 결과를 확인하기 위한 데스크톱 UI (Python 표준 라이브러리인 tkinter로만
# 작성해서 별도 설치 없이 바로 실행 가능하다).
#
# 화면 구성:
#   1) [사진 선택] 버튼         : 테스트할 얼굴 사진을 여러 장 한 번에 선택할 수 있다
#                                (Ctrl/Shift 클릭으로 다중 선택). 화면에는 그중 대표로
#                                1장만 계속 표시하지만, 측정 자체는 선택한 사진 전부에 대해 이뤄진다.
#   2) 사진 미리보기            : 대표 사진 1장을 화면에 계속 띄워서 보여준다 (버튼을 눌러도 유지됨).
#   3) [안 1 / 안 2 / 비교] 버튼 : 어떤 학습 결과(모델)로 매력도를 볼지 고른다 (기존과 동일하게 유지).
#   4) 결과 표(Table)          : 대표 사진의 예측 점수/등급/얼굴 인식 여부/참고 성능/주요 판단
#                                부위를 한눈에 비교할 수 있도록 표(ttk.Treeview)로 정리해서 보여준다.
#   5) Grad-CAM 히트맵          : 대표 사진에 대해 "모델이 얼굴의 어느 부위를 보고 이 점수를
#                                줬는지"를 보여주는 히트맵 이미지와 설명 문구를 모델별로 보여준다.
#   6) [HTML로 저장] 버튼       : 선택했던 사진 전부의 측정 결과(사진 + 예측 점수 + 데이터셋에
#                                있는 사진이면 실제 점수까지)를 카드 형태의 HTML 파일 하나로 저장한다.
#   7) 하단 상태창              : 얼굴 인식 여부/진행 상황/오류 등 짧은 안내 메시지를 출력한다.
#
# 실행 전 준비:
#   먼저 train_aihub.py 와 train_all.py 를 각각 실행해서
#   outputs/aihub/best_model_aihub.pt , outputs/all/best_model_all.pt 체크포인트를 만들어 두어야 한다.
#
# 실행 방법:
#   python app_gui.py
# ------------------------------------------------------------------------------------

import os
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, scrolledtext, ttk

import torch
from PIL import Image, ImageTk

# 이 파일(DenseNet121/app_gui.py)을 "python app_gui.py"로 직접 실행하면 파이썬은
# 이 파일이 있는 폴더(DenseNet121/)만 sys.path에 자동으로 넣어준다. training/, inference/,
# gui/ 하위 패키지를 곧바로 import할 수 있도록 그 경로가 sys.path에 없다면 추가해 둔다.
# (반대로 "python -m DenseNet121.app_gui"처럼 패키지로 실행했을 때도 동일하게 동작한다.)
_DENSENET121_DIR = os.path.dirname(os.path.abspath(__file__))
if _DENSENET121_DIR not in sys.path:
    sys.path.insert(0, _DENSENET121_DIR)

from gui.html_report import build_html_report
from inference.predict import predict_one, lookup_ground_truth, MODEL_LABELS
from training.train_common import DEFAULT_OUTPUT_ROOT

# images/, labels_combined_1to5.xlsx 와 마찬가지로 outputs/ 도 DenseNet121 코드 폴더가 아니라
# 프로젝트 루트에 있다 (train_aihub.py/train_all.py 의 기본 저장 위치와 동일해야 함).
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

PREVIEW_SIZE = (260, 260)   # 업로드한 원본 사진 미리보기 크기
GRADCAM_SIZE = (200, 200)   # 모델별 Grad-CAM 히트맵 썸네일 크기

# 결과 표(Treeview)의 컬럼 정의: (내부 키, 화면에 보일 제목, 폭)
TABLE_COLUMNS = [
    ("model", "모델", 170),
    ("score", "예측 점수", 80),
    ("grade", "등급", 50),
    ("face", "얼굴 인식", 90),
    ("mae", "참고 MAE", 80),
    ("acc1", "참고 정확도(±1)", 100),
    ("focus", "주요 판단 부위", 130),
]


class AttractivenessApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("얼굴 매력도 예측기 (DenseNet121)")
        root.geometry("760x860")

        # GPU(CUDA)가 있으면 자동으로 사용하고, 없으면 CPU를 사용한다.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 여러 장을 선택할 수 있으므로 전체 목록(image_paths)을 들고 있되, 화면에는
        # 그중 대표 1장(image_path, 항상 목록의 첫 번째)만 계속 표시한다.
        self.image_paths: list[str] = []
        self.image_path: str | None = None

        # [HTML로 저장] 버튼이 사용할, 가장 최근 측정의 전체 결과.
        # {image_path: {"dataset":.., "filename":.., "actual_score":.., "models": {run_name: 결과dict}}}
        self.batch_results: dict = {}
        self.last_run_names: list[str] = []

        # tkinter는 PhotoImage를 변수에 붙잡아두지 않으면 가비지 컬렉션으로 사라져 화면에서
        # 사라져버리므로, 현재 표시 중인 이미지들을 아래 속성들에 계속 참조로 들고 있는다.
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._gradcam_photos: list[ImageTk.PhotoImage] = []

        # --- 0) 스크롤 가능한 컨테이너 -------------------------------------------------
        # 위젯이 많아 창 높이보다 내용이 길어지면 아래쪽이 잘려서 안 보일 수 있으므로,
        # 전체 내용을 캔버스+스크롤바로 감싸서 마우스 휠이나 스크롤바로 스크롤할 수 있게 한다.
        # (이 아래의 모든 위젯은 root가 아니라 이 content 프레임을 부모로 삼는다.)
        outer_canvas = tk.Canvas(root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=scrollbar.set)
        outer_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = tk.Frame(outer_canvas)
        canvas_window = outer_canvas.create_window((0, 0), window=content, anchor="nw")

        # 내용 프레임의 크기가 바뀔 때마다(위젯 추가/제거 포함) 스크롤 범위를 다시 계산한다.
        content.bind(
            "<Configure>",
            lambda e: outer_canvas.configure(scrollregion=outer_canvas.bbox("all")),
        )
        # 창 폭이 바뀌면 내용 프레임도 같은 폭으로 맞춰서 좌우로 잘리거나 남는 공백이 없게 한다.
        outer_canvas.bind("<Configure>", lambda e: outer_canvas.itemconfig(canvas_window, width=e.width))

        # 마우스 휠로도 스크롤할 수 있게 한다 (Windows 기준 <MouseWheel>, delta는 120의 배수).
        def _on_mousewheel(event):
            outer_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        outer_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- 1) 사진 경로 표시 텍스트 상자 + 사진 선택 버튼 ---------------------------
        top_frame = tk.Frame(content)
        top_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.path_entry = tk.Entry(top_frame)
        self.path_entry.pack(side="left", fill="x", expand=True)
        self._set_path_entry_text("선택된 사진 없음")

        select_btn = tk.Button(top_frame, text="사진 선택(여러 장 가능)", width=18, command=self.on_select_image)
        select_btn.pack(side="left", padx=(8, 0))

        # --- 2) 대표 사진을 계속 보여주는 미리보기 영역 --------------------------------
        preview_frame = tk.Frame(content)
        preview_frame.pack(pady=5)
        self.preview_label = tk.Label(
            preview_frame, text="선택된 사진 없음", width=34, height=13,
            bg="#dddddd", relief="groove",
        )
        self.preview_label.pack()

        # --- 3) 어떤 모델(안)로 결과를 볼지 고르는 버튼들 (기존 기능 그대로 유지) --------
        btn_frame = tk.Frame(content)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.model_buttons = [
            tk.Button(btn_frame, text="안 1: AIHub 모델 결과", command=lambda: self.on_predict("aihub")),
            tk.Button(btn_frame, text="안 2: 전체 데이터 모델 결과", command=lambda: self.on_predict("all")),
            tk.Button(btn_frame, text="두 모델 비교", command=lambda: self.on_predict("both")),
        ]
        for b in self.model_buttons:
            b.pack(side="left", expand=True, fill="x", padx=4)

        # --- 4) 예측 결과를 한눈에 비교하는 표(Table, 대표 사진 기준) -------------------
        table_frame = tk.Frame(content)
        table_frame.pack(fill="x", padx=10, pady=(10, 5))

        col_keys = [key for key, _, _ in TABLE_COLUMNS]
        self.tree = ttk.Treeview(table_frame, columns=col_keys, show="headings", height=3)
        for key, title, width in TABLE_COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="center")
        self.tree.pack(side="left", fill="x", expand=True)

        # --- 5) 모델별 Grad-CAM 히트맵(점수 판단 근거, 대표 사진 기준) 표시 영역 ---------
        gradcam_outer = tk.LabelFrame(content, text="점수 판단 근거 (Grad-CAM 히트맵, 대표 사진 기준)")
        gradcam_outer.pack(fill="x", padx=10, pady=5)
        self.gradcam_frame = tk.Frame(gradcam_outer)
        self.gradcam_frame.pack(fill="x", padx=5, pady=5)
        self._render_empty_gradcam_placeholder()

        # --- 6) 전체 측정 결과를 HTML로 저장하는 버튼 ---------------------------------
        save_frame = tk.Frame(content)
        save_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.save_html_btn = tk.Button(
            save_frame, text="HTML로 저장 (선택한 사진 전체 결과)", command=self.on_save_html, state="disabled",
        )
        self.save_html_btn.pack(fill="x")

        # --- 7) 상태/오류/진행 메시지를 출력하는 작은 텍스트 상자 -----------------------
        self.status_box = scrolledtext.ScrolledText(content, height=8, state="disabled", wrap="word")
        self.status_box.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.log(f"사용 장치: {self.device}")
        self.log(f"모델 체크포인트 폴더: {OUTPUT_ROOT}")
        self.log("사용법: [사진 선택]으로 얼굴 사진을 (여러 장도 가능) 고른 뒤, 원하는 모델 버튼을 눌러주세요.")

    # ------------------------------------------------------------------ 화면 갱신 유틸
    def _set_path_entry_text(self, text: str):
        self.path_entry.config(state="normal")
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, text)
        self.path_entry.config(state="readonly")

    def log(self, text: str):
        """상태 텍스트 상자 맨 아래에 한 줄(들)을 추가한다."""
        self.status_box.config(state="normal")
        self.status_box.insert("end", text + "\n")
        self.status_box.see("end")  # 항상 최신 로그가 보이도록 자동 스크롤
        self.status_box.config(state="disabled")

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in self.model_buttons:
            b.config(state=state)

    def _render_empty_gradcam_placeholder(self):
        for widget in self.gradcam_frame.winfo_children():
            widget.destroy()
        tk.Label(self.gradcam_frame, text="아직 예측 결과가 없습니다.").pack(padx=10, pady=20)

    def _show_preview_image(self, pil_image: Image.Image):
        """대표 사진을 미리보기 영역에 표시한다 (비율 유지, 잘리지 않게 축소)."""
        thumb = pil_image.copy()
        thumb.thumbnail(PREVIEW_SIZE)
        self._preview_photo = ImageTk.PhotoImage(thumb)
        self.preview_label.config(image=self._preview_photo, text="", width=thumb.width, height=thumb.height)

    # ------------------------------------------------------------------ 버튼 콜백
    def on_select_image(self):
        # askopenfilenames (복수형): Ctrl/Shift 클릭으로 여러 장을 한 번에 선택할 수 있다.
        paths = filedialog.askopenfilenames(
            title="테스트할 얼굴 사진 선택 (여러 장 선택 가능)",
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")],
        )
        if not paths:  # 사용자가 파일 선택을 취소한 경우
            return
        self.image_paths = list(paths)
        self.image_path = self.image_paths[0]  # 화면에 보여줄 대표 사진

        if len(self.image_paths) == 1:
            self._set_path_entry_text(self.image_path)
        else:
            self._set_path_entry_text(f"{len(self.image_paths)}장 선택됨 (대표: {os.path.basename(self.image_path)})")

        # 대표 사진만 화면에 계속 띄워서, 버튼을 눌러도 "지금 어떤 사진을 보고 있는지" 계속 보이게 한다.
        self._show_preview_image(Image.open(self.image_path).convert("RGB"))
        self.log(f"\n{len(self.image_paths)}장의 사진을 선택했습니다. (대표 표시: {self.image_path})")

    def on_predict(self, model_choice: str):
        """[안 1] / [안 2] / [두 모델 비교] 버튼이 눌렸을 때 실행된다."""
        if not self.image_paths:
            self.log("[안내] 먼저 [사진 선택] 버튼으로 사진을 선택해주세요.")
            return

        self._set_buttons_enabled(False)
        self.save_html_btn.config(state="disabled")
        self.log(f"\n총 {len(self.image_paths)}장 측정을 시작합니다... "
                 f"(대표 사진만 Grad-CAM까지 계산하며, 나머지는 점수만 빠르게 계산합니다)")

        # 모델 로딩/추론에 시간이 걸려도 창이 멈춘 것처럼 보이지 않도록 별도 스레드에서 실행한다.
        thread = threading.Thread(target=self._run_predict_thread, args=(model_choice,), daemon=True)
        thread.start()

    def on_save_html(self):
        if not self.batch_results:
            self.log("[안내] 먼저 모델 버튼([안 1]/[안 2]/[두 모델 비교])을 눌러 측정을 실행해주세요.")
            return

        save_path = filedialog.asksaveasfilename(
            title="측정 결과 HTML로 저장",
            defaultextension=".html",
            filetypes=[("HTML 파일", "*.html")],
            initialfile="attractiveness_report.html",
        )
        if not save_path:
            return

        html_content = build_html_report(self.batch_results, self.last_run_names)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.log(f"\nHTML로 저장했습니다: {save_path}")
        try:
            webbrowser.open("file://" + save_path)
        except Exception:
            pass  # 브라우저 자동 실행에 실패해도 파일 저장 자체는 이미 완료된 상태이므로 무시한다.

    # ------------------------------------------------------------------ 백그라운드 작업
    def _run_predict_thread(self, model_choice: str):
        run_names = ["aihub", "all"] if model_choice == "both" else [model_choice]
        total = len(self.image_paths)
        batch_results: dict = {}
        representative_items = []  # 화면(표/Grad-CAM)에 보여줄 대표 사진의 (run_name, result, error) 목록

        for idx, image_path in enumerate(self.image_paths, start=1):
            is_representative = (image_path == self.image_path)
            dataset, filename, actual_score = lookup_ground_truth(image_path)
            item = {"dataset": dataset, "filename": filename, "actual_score": actual_score, "models": {}}

            for run_name in run_names:
                try:
                    # 대표 사진만 Grad-CAM(히트맵+판단 근거)까지 계산하고, 나머지는 점수만 빠르게 계산한다.
                    result = predict_one(
                        image_path, run_name, self.device, OUTPUT_ROOT, with_gradcam=is_representative,
                    )
                    item["models"][run_name] = result
                    if is_representative:
                        representative_items.append((run_name, result, None))
                except Exception as e:  # 체크포인트 없음, 얼굴 검출 오류, 손상된 이미지 파일 등
                    if is_representative:
                        representative_items.append((run_name, None, e))
                    else:
                        # 대표 사진이 아니어도 실패 사실 자체는 반드시 로그에 남겨서 조용히 묻히지 않게 한다.
                        label = MODEL_LABELS.get(run_name, run_name)
                        self.root.after(
                            0, self.log,
                            f"  [오류] {os.path.basename(image_path)} / [{label}] {e}",
                        )

            batch_results[image_path] = item
            # tkinter 위젯은 메인 스레드에서만 안전하게 갱신할 수 있으므로 root.after로 예약한다.
            self.root.after(0, self.log, f"  [{idx}/{total}] 측정 완료: {os.path.basename(image_path)}")

        self.root.after(0, self._on_predict_done, representative_items, batch_results, run_names)

    def _on_predict_done(self, items, batch_results: dict, run_names: list[str]):
        # 이전 결과를 지우고 대표 사진의 새 결과로 표와 Grad-CAM 영역을 다시 채운다.
        self.tree.delete(*self.tree.get_children())
        for widget in self.gradcam_frame.winfo_children():
            widget.destroy()
        self._gradcam_photos.clear()

        success_count = 0
        for run_name, result, error in items:
            if error is not None:
                label = MODEL_LABELS.get(run_name, run_name)
                self.log(f"[오류] [{label}] {error}")
                continue

            success_count += 1
            if result["face_found"] is True:
                self.log(f"[정보] ({run_name}) 얼굴을 인식하여 해당 영역으로 잘라냈습니다.")
            elif result["face_found"] is False:
                self.log(f"[경고] ({run_name}) 얼굴을 찾지 못했습니다. 이미지 전체를 사용해 예측했습니다.")

            self._insert_table_row(run_name, result)
            self._add_gradcam_panel(run_name, result)

        if success_count == 0:
            self._render_empty_gradcam_placeholder()

        # HTML 저장 버튼이 사용할 전체(여러 장) 결과를 저장해둔다.
        self.batch_results = batch_results
        self.last_run_names = run_names
        self.log(f"\n총 {len(batch_results)}장 측정 완료. [HTML로 저장] 버튼으로 전체 결과를 파일로 저장할 수 있습니다.")

        self._set_buttons_enabled(True)
        self.save_html_btn.config(state="normal")

    def _insert_table_row(self, run_name: str, result: dict):
        ref = result["reference_metrics"]
        mae_text = f"{ref['mae']:.3f}" if ref is not None else "N/A"
        acc1_text = f"{ref['accuracy_within1']*100:.1f}%" if ref is not None else "N/A"
        face_text = {True: "인식됨", False: "미인식(전체 사용)"}.get(result["face_found"], "-")

        self.tree.insert("", "end", values=(
            MODEL_LABELS.get(run_name, run_name),
            f"{result['score']:.2f} / 5.00",
            f"{result['rounded']}점",
            face_text,
            mae_text,
            acc1_text,
            result["reason_short"],
        ))

    def _add_gradcam_panel(self, run_name: str, result: dict):
        """모델 하나의 Grad-CAM 히트맵 이미지 + 설명 문구를 gradcam_frame에 나란히 추가한다."""
        panel = tk.Frame(self.gradcam_frame, bd=1, relief="solid")
        panel.pack(side="left", padx=8, pady=5, expand=True, fill="both")

        tk.Label(panel, text=MODEL_LABELS.get(run_name, run_name), font=("", 9, "bold")).pack(pady=(4, 2))

        thumb = result["gradcam_image"].copy()
        thumb.thumbnail(GRADCAM_SIZE)
        photo = ImageTk.PhotoImage(thumb)
        self._gradcam_photos.append(photo)  # 가비지 컬렉션 방지를 위해 계속 참조 유지
        tk.Label(panel, image=photo).pack(padx=6)

        tk.Label(
            panel, text=result["reason_text"], wraplength=GRADCAM_SIZE[0] + 20,
            justify="left", fg="#333333",
        ).pack(padx=6, pady=(4, 8))


def main():
    root = tk.Tk()
    AttractivenessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
