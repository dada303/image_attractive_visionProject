// Future model integration point. No network requests or placeholder scores.
export const modelInfo = Object.freeze({ name: "DenseNet121", ready: false });
export async function predictImage(_image) {
  throw new Error("DenseNet 모델이 아직 연결되지 않았습니다.");
}
