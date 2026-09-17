import { cropImage, predictImage } from "./model.js";
const $ = (id) => document.getElementById(id);
const fileInput = $("file"), photo = $("photo"), dialog = $("camera-dialog"), video = $("video");
let selectedBlob = null, cropToken = null, busy = false, modelsReady = false;
let cropController = null, predictionController = null, modelVersion = 0;
let imageUrl = null, stream = null, version = 0, cameraVersion = 0, selectedName = "";
const MAX_BYTES = 10 * 1024 * 1024, MAX_PIXELS = 20_000_000;
function status(message = "") { $("status").textContent = message; }
function resetResult() {
  $('result').hidden = true;
  $('score').textContent = '—';
  $('score-meter').value = 1; $('result-model').textContent = '';
}
function clearPhoto() {
  version++;
  cropController?.abort(); predictionController?.abort();
  cropController = null; predictionController = null;
  selectedBlob = null; cropToken = null; busy = false; resetResult();
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null; photo.removeAttribute('src'); selectedName = '';
  $('filename').textContent = ''; $('dimensions').textContent = '';
  $('hero').hidden = false; $('photo-panel').hidden = true;
  $('analyze').disabled = true; $('analyze').textContent = '결과 보기';
  $('download').disabled = true;
  $('upload-label').textContent = '사진 선택하기'; fileInput.value = ''; status();
}
async function selectPhoto(blob, name) {
  clearPhoto();
  const token = version;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(blob.type)) { status('JPG, PNG, WebP 사진을 선택해주세요.'); return; }
  if (!blob.size || blob.size > MAX_BYTES) { status('10MB 이하의 사진을 선택해주세요.'); return; }
  cropController = new AbortController();
  status('얼굴 인식 중…');
  let url = null;
  try {
    const crop = await cropImage(blob, name, cropController.signal);
    if (token !== version) return;
    url = URL.createObjectURL(crop.blob);
    const probe = new Image(); probe.src = url; await probe.decode();
    if (token !== version) { URL.revokeObjectURL(url); return; }
    // This exact lossless PNG blob is BOTH displayed and posted to predict.
    selectedBlob = crop.blob; cropToken = crop.token;
    imageUrl = url; selectedName = name; photo.src = imageUrl;
    $('filename').textContent = name;
    $('dimensions').textContent = probe.naturalWidth + ' × ' + probe.naturalHeight;
    $('hero').hidden = true; $('photo-panel').hidden = false;
    $('upload-label').textContent = '다른 사진 선택하기';
    $('analyze').disabled = !modelsReady; $('download').disabled = false;
    status((crop.faceCount > 1 ? '여러 얼굴 중 가장 큰 얼굴을 선택했습니다. ' : '') + 'Crop을 확인한 후 결과 보기를 눌러주세요.');
  } catch (error) {
    if (url) URL.revokeObjectURL(url);
    if (token === version && error.name !== 'AbortError') status(error.message);
  }
}
$("upload").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0]; if (!file) return;
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
  clearPhoto();
  const token = version;
  $("capture").disabled = true;
  canvas.toBlob((blob) => {
    dialog.close();
    if (blob && token === version) selectPhoto(blob, "camera-" + Date.now() + ".png");
  }, "image/png");
});
$('download').addEventListener('click', () => {
  if (!imageUrl || !selectedBlob) return;
  const a = document.createElement('a');
  a.href = imageUrl; a.download = 'face-crop-' + (selectedName.replace(/\.[^.]+$/, '') || 'photo') + '.png';
  document.body.append(a); a.click(); a.remove();
  status('확인한 Crop 이미지를 PNG로 저장했습니다.');
});
window.addEventListener("pagehide", () => { stopCamera(); if (imageUrl) URL.revokeObjectURL(imageUrl); });
document.addEventListener("visibilitychange", () => { if (document.hidden && dialog.open) dialog.close(); });

$('analyze').addEventListener('click', async () => {
  if (!selectedBlob || busy || !modelsReady) return;
  const token = version, currentModelVersion = modelVersion;
  predictionController = new AbortController();
  const modelId = $('model-select').value;
  const modelLabel = $('model-select').selectedOptions[0].textContent;
  busy = true; $('analyze').disabled = true;
  $('analyze').textContent = '분석 중…'; resetResult();
  status(modelLabel + ' 모델이 사진을 분석하고 있어요…');
  try {
    const result = await predictImage(selectedBlob, modelId, cropToken, predictionController.signal);
    if (token !== version || currentModelVersion !== modelVersion) return;
    $('score').textContent = result.score.toFixed(2);
    $('score-meter').value = result.score;
    $('result-model').textContent = result.model + ' · AI 예측 점수';
    $('result').hidden = false;
    status('분석 완료');
  } catch (error) {
    if (token === version && currentModelVersion === modelVersion && error.name !== 'AbortError') status(error.message);
  } finally {
    if (token === version && currentModelVersion === modelVersion) {
      busy = false; $('analyze').disabled = !selectedBlob || !modelsReady;
      $('analyze').textContent = '결과 보기';
    }
  }
});

$('model-select').addEventListener('change', () => {
  modelVersion++; predictionController?.abort(); busy = false; resetResult();
  $('analyze').disabled = !selectedBlob || !modelsReady; $('analyze').textContent = '결과 보기';
  $('model-badge').textContent = $('model-select').selectedOptions[0].textContent;
  status(selectedBlob ? '모델 변경 완료 · 결과 보기를 눌러주세요.' : '');
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
