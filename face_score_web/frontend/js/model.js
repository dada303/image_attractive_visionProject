export async function predictImage(blob, modelId) {
  let response;
  try {
    response = await fetch('/api/predict?model=' + encodeURIComponent(modelId), {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
      body: blob, signal: AbortSignal.timeout(120000),
    });
  } catch (error) {
    throw new Error(error.name === 'TimeoutError' ? '예측 시간이 초과되었습니다. 다시 시도해주세요.' : '로컬 프로그램 연결을 확인해주세요.');
  }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '점수 예측에 실패했습니다.');
  if (!Number.isFinite(result.score)) throw new Error('유효한 점수가 아닙니다.');
  return result;
}
