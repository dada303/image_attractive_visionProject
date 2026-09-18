async function requestImage(route, blob, filename, { signal, token } = {}) {
  let response;
  try {
    response = await fetch(route, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(filename), ...(token ? { 'X-Crop-Token': token } : {}) },
      body: blob,
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(120000)]) : AbortSignal.timeout(120000),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error(error.name === 'TimeoutError' ? '처리 시간이 초과되었습니다. 다시 시도해주세요.' : '프로그램 연결을 확인해주세요.');
  }
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.error || '이미지 처리에 실패했습니다.');
  }
  return response;
}

export async function cropImage(blob, filename, signal) {
  const response = await requestImage('/api/crop', blob, filename, { signal });
  const token = response.headers.get('X-Crop-Token');
  const cropped = await response.blob();
  if (!token || cropped.type !== 'image/png') throw new Error('올바른 Crop 응답이 아닙니다.');
  return { blob: cropped, token, faceCount: Number(response.headers.get('X-Face-Count') || 1) };
}

export async function predictImage(blob, modelId, token, signal) {
  const response = await requestImage('/api/predict?model=' + encodeURIComponent(modelId), blob, 'face-crop.png', { token, signal });
  const result = await response.json();
  if (!Number.isFinite(result.score)) throw new Error('유효한 점수가 아닙니다.');
  return result;
}
