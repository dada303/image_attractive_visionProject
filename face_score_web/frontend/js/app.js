import { predictImage } from "./model.js";
const $ = (id) => document.getElementById(id);
const fileInput = $("file"), photo = $("photo"), dialog = $("camera-dialog"), video = $("video");
let selectedBlob = null, busy = false, modelsReady = false;
let imageUrl = null, stream = null, version = 0, cameraVersion = 0, selectedName = "";
const MAX_BYTES = 10 * 1024 * 1024, MAX_PIXELS = 20_000_000;
function status(message = "") { $("status").textContent = message; }
function resetResult() {
  $('result').hidden = true;
  $('score').textContent = '—';
}
function clearPhoto() {
  selectedBlob = null; resetResult();
  version++;
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null; photo.removeAttribute("src"); selectedName = "";
  $("hero").hidden = false; $("photo-panel").hidden = true; $("selected-actions").hidden = true;
  $("upload-label").textContent = "사진 선택하기"; fileInput.value = ""; status();
}
async function selectPhoto(blob, name) {
  const token = ++version;
  selectedBlob = null; resetResult(); $('analyze').disabled = true;
  status("사진을 확인하고 있어요…");
  if (!blob.size || blob.size > MAX_BYTES) { status("10MB 이하의 사진을 선택해주세요."); return; }
  const url = URL.createObjectURL(blob), probe = new Image();
  try {
    probe.src = url; await probe.decode();
    if (token !== version) { URL.revokeObjectURL(url); return; }
    if (!probe.naturalWidth || probe.naturalWidth * probe.naturalHeight > MAX_PIXELS) throw new Error("사진이 너무 커요. 2,000만 화소 이하로 줄여주세요.");
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    selectedBlob = blob; imageUrl = url; selectedName = name; photo.src = url;
    $("filename").textContent = name;
    $("dimensions").textContent = probe.naturalWidth + " × " + probe.naturalHeight;
    $("hero").hidden = true; $("photo-panel").hidden = false; $("selected-actions").hidden = false;
    $("upload-label").textContent = "다른 사진 선택하기";
    $("analyze").disabled = busy || !modelsReady;
    status("사진 준비 완료 · 점수 확인을 눌러주세요.");
  } catch (error) {
    URL.revokeObjectURL(url);
    if (token === version) status(error.message?.includes("화소") ? error.message : "사진을 읽을 수 없어요. JPG, PNG 또는 WebP 파일로 다시 선택해주세요.");
  }
}
$("upload").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0]; if (!file) return;
  if (!["image/jpeg","image/png","image/webp"].includes(file.type)) { status("JPG, PNG, WebP 사진을 선택해주세요."); fileInput.value = ""; return; }
  selectPhoto(file, file.name); fileInput.value = "";
});
$("remove").addEventListener("click", clearPhoto);
function stopCamera() {
  cameraVersion++;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null; video.srcObject = null; $("capture").disabled = true;
}
$("camera").addEventListener("click", async () => {
  status();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    status("카메라는 HTTPS 또는 localhost에서 사용할 수 있어요. 사진 선택 기능을 이용해주세요."); return;
  }
  stopCamera(); const token = cameraVersion;
  dialog.showModal(); $("camera-status").textContent = "카메라 권한을 허용해주세요.";
  try {
    const incoming = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 1280 } }, audio: false });
    if (token !== cameraVersion || !dialog.open) { incoming.getTracks().forEach(t => t.stop()); return; }
    stream = incoming; video.srcObject = stream; await video.play();
    if (token !== cameraVersion || !dialog.open) return;
    $("capture").disabled = false; $("camera-status").textContent = "얼굴을 중앙에 두고 촬영해주세요.";
  } catch (error) {
    if (token !== cameraVersion) return;
    stopCamera();
    $("camera-status").textContent = error.name === "NotAllowedError" ? "카메라 권한이 필요해요. 권한을 허용하거나 사진을 선택해주세요." : "카메라를 사용할 수 없어요. 연결 상태를 확인하거나 사진을 선택해주세요.";
  }
});
$("close-camera").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", stopCamera);
dialog.addEventListener("cancel", stopCamera);
$("capture").addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) return;
  const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const token = version;
  $("capture").disabled = true;
  canvas.toBlob((blob) => {
    dialog.close();
    if (blob && token === version) selectPhoto(blob, "camera-" + Date.now() + ".jpg");
  }, "image/jpeg", .92);
});
$("download").addEventListener("click", () => {
  if (!imageUrl || !photo.naturalWidth) return;
  const canvas = document.createElement("canvas"); const ratio = Math.min(1, 1600 / Math.max(photo.naturalWidth, photo.naturalHeight));
  canvas.width = Math.round(photo.naturalWidth * ratio); canvas.height = Math.round(photo.naturalHeight * ratio);
  canvas.getContext("2d").drawImage(photo, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (!blob) { status("이미지 저장에 실패했어요. 다시 시도해주세요."); return; }
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = "face-note-" + (selectedName.replace(/\.[^.]+$/, "") || "photo") + ".png";
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    status("사진을 PNG로 저장했습니다.");
  }, "image/png");
});
window.addEventListener("pagehide", () => { stopCamera(); if (imageUrl) URL.revokeObjectURL(imageUrl); });
document.addEventListener("visibilitychange", () => { if (document.hidden && dialog.open) dialog.close(); });

$('analyze').addEventListener('click', async () => {
  if (!selectedBlob || busy || !modelsReady) return;
  const token = version;
  const modelId = $('model-select').value;
  const modelLabel = $('model-select').selectedOptions[0].textContent;
  busy = true; $('analyze').disabled = true;
  $('analyze').textContent = '분석 중…'; resetResult();
  status(modelLabel + ' 모델이 사진을 분석하고 있어요…');
  try {
    const result = await predictImage(selectedBlob, modelId, selectedName);
    if (token !== version) return;
    $('score').textContent = result.score.toFixed(2);
    $('score-meter').value = result.score;
    $('result-model').textContent = result.model + ' · AI 예측 점수';
    $('result').hidden = false;
    status('분석 완료');
  } catch (error) {
    if (token === version) status(error.message);
  } finally {
    busy = false; $('analyze').disabled = !selectedBlob || !modelsReady;
    $('analyze').textContent = '점수 확인';
  }
});

$('model-select').addEventListener('change', () => {
  version++; resetResult();
  $('model-badge').textContent = $('model-select').selectedOptions[0].textContent;
  status(selectedBlob ? '모델 변경 완료 · 점수 확인을 눌러주세요.' : '');
});
async function loadModels() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error();
    const data = await response.json();
    const select = $('model-select'); select.replaceChildren();
    for (const model of data.models) {
      const option = new Option(model.label + (model.ready ? '' : ' (사용 불가)'), model.id);
      option.disabled = !model.ready; select.add(option);
    }
    const first = data.models.find(model => model.ready && model.id === data.default_model) || data.models.find(model => model.ready);
    if (!first) throw new Error();
    select.value = first.id; select.disabled = false; modelsReady = true;
    $('model-badge').textContent = first.label;
    $('analyze').disabled = !selectedBlob || busy;
  } catch {
    status('모델 목록을 불러오지 못했습니다. 로컬 프로그램을 확인하고 새로고침해주세요.');
  }
}
loadModels();
