"""FastAPI 전환 후에도 기존 HTTP·서비스 계약이 유지되는지 검증한다."""

from pathlib import Path
from threading import Event
import asyncio
import hashlib
import hmac
import httpx
import os
import sys
import time
import unittest
from unittest.mock import patch

WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.common.errors import CropError
from backend.common.request_validation import validate_upload
from backend.config import FRONTEND, MAX_BYTES
from backend.services.crop_service import CropService
from backend.services.model_service import ModelRegistry


class FakeCrops:
    """실제 얼굴 검출 없이 Crop 서비스의 반환 계약을 재현한다."""

    def prepare(self, data):
        """입력 이미지 대신 고정 PNG 본문·토큰·얼굴 수를 반환한다."""
        return b"cropped-image", "crop-token", {"face_count": 1}


class FakeRegistry:
    """컨트롤러가 의존하는 모델 Registry 계약을 재현한다."""

    def __init__(self):
        """로컬·운영 기본 모델과 테스트용 실행 제어 상태를 준비한다."""
        self.scorers = {"densenet121": object(), "efficientnet_b0": object()}
        self.crops = FakeCrops()
        self.fail_next_prediction = False
        self.block_predictions = False
        self.prediction_entered = Event()
        self.release_prediction = Event()

    def list_models(self):
        """가용 모델 API에 노출할 고정 목록을 반환한다."""
        return [
            {"id": "densenet121", "label": "DenseNet121", "ready": True, "error": None},
            {"id": "efficientnet_b0", "label": "EfficientNet-B0", "ready": True, "error": None},
        ]

    def predict_crop(self, data, token, model_id):
        """요청 인자를 반환하며 테스트 설정에 따라 실패하거나 실행을 대기한다."""
        if self.fail_next_prediction:
            self.fail_next_prediction = False
            raise RuntimeError("temporary inference failure")
        if self.block_predictions:
            self.prediction_entered.set()
            if not self.release_prediction.wait(timeout=5):
                raise TimeoutError("test did not release prediction")
        return {"model_id": model_id, "token": token, "size": len(data), "score": 3.5}


class ImmediateTimeout:
    """업로드 대기 없이 asyncio timeout 발생을 재현하는 비동기 컨텍스트 관리자."""

    async def __aenter__(self):
        """본문 스트림을 읽기 전에 TimeoutError를 발생시킨다."""
        raise TimeoutError

    async def __aexit__(self, exception_type, exception, traceback):
        """예외를 숨기지 않고 요청 처리부로 전달한다."""
        return False


class FastAPIContractCase(unittest.TestCase):
    """공통 FastAPI 요청 도우미를 제공한다."""

    base_url = "http://localhost:8767"
    client: TestClient

    def post(self, path: str, data: bytes = b"image", token: str = "", **headers: str):
        """기존 이미지 바이너리 헤더를 붙여 API에 POST한다."""
        request_headers: dict[str, str] = {
            "Content-Type": "application/octet-stream",
            "X-Filename": "photo.png",
            "X-Crop-Token": token,
        }
        request_headers.update(headers)
        return self.client.post(path, content=data, headers=request_headers)


class ProductionContractTests(FastAPIContractCase):
    """운영 FastAPI 앱이 기존 Flask HTTP 계약을 유지하는지 검증한다."""

    def setUp(self):
        """각 테스트에 독립 Registry와 운영 FastAPI 클라이언트를 만든다."""
        self.registry = FakeRegistry()
        environment = {
            "PUBLIC_ORIGIN": "",
            "PORT": "8767",
            "DEFAULT_MODEL": "efficientnet_b0",
            "ENABLED_MODELS": "densenet121,efficientnet_b0",
        }
        with patch.dict(os.environ, environment, clear=False):
            self.app = create_app(self.registry)
        self.client = TestClient(self.app, base_url=self.base_url)

    def test_health_and_models_expose_registry_state(self):
        """상태 API가 운영 기본 모델과 가용 목록을 기존 형태로 반환하는지 확인한다."""
        for path in ("/api/health", "/api/models"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ready"])
            self.assertEqual(response.json()["default_model"], "efficientnet_b0")
            self.assertEqual(response.json()["models"], self.registry.list_models())

    def test_head_options_and_health_failure_preserve_framework_contracts(self):
        """Flask의 HEAD·OPTIONS 동작과 상태 서비스 예외의 공통 500 응답을 유지하는지 확인한다."""
        head = self.client.head("/api/health")
        health_options = self.client.options("/api/health")
        predict_options = self.client.options("/api/predict")
        with patch.object(self.registry, "list_models", side_effect=RuntimeError("registry unavailable")):
            failed = self.client.get("/api/health")

        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(health_options.status_code, 200)
        self.assertEqual(health_options.headers["Allow"], "GET, HEAD, OPTIONS")
        self.assertEqual(predict_options.status_code, 200)
        self.assertEqual(predict_options.headers["Allow"], "GET, HEAD, OPTIONS, POST")
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json(), {"error": "요청을 처리할 수 없습니다."})
        self.assertEqual(failed.headers["X-Content-Type-Options"], "nosniff")

    def test_crop_and_predict_preserve_response_contracts(self):
        """Crop 본문·헤더와 모델·토큰을 전달하는 예측 JSON 계약을 확인한다."""
        crop = self.post("/api/crop")
        self.assertEqual(crop.status_code, 200)
        self.assertEqual(crop.content, b"cropped-image")
        self.assertEqual(crop.headers["X-Crop-Token"], "crop-token")
        self.assertEqual(crop.headers["X-Face-Count"], "1")

        prediction = self.post(
            "/api/predict?model=efficientnet_b0",
            data=crop.content,
            token=crop.headers["X-Crop-Token"],
        )
        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(prediction.json()["model_id"], "efficientnet_b0")
        self.assertEqual(prediction.json()["token"], "crop-token")
        self.assertEqual(prediction.json()["size"], len(crop.content))

    def test_host_origin_and_security_headers_are_preserved(self):
        """운영 Host·Origin 차단과 기존 보안 응답 헤더를 확인한다."""
        bad_host = TestClient(self.app, base_url="http://example.test").get("/api/health")
        bad_origin = self.client.get("/api/health", headers={"Origin": "https://example.test"})
        response = self.client.get("/api/health")

        self.assertEqual(bad_host.status_code, 403)
        self.assertEqual(bad_host.json(), {"error": "허용되지 않은 접속 주소입니다. PUBLIC_ORIGIN 설정을 확인해주세요."})
        self.assertEqual(bad_origin.status_code, 403)
        self.assertEqual(bad_origin.json(), {"error": "허용되지 않은 요청 출처입니다."})
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_public_origin_accepts_cloudflare_host_without_trusting_forwarded_host(self):
        """공개 HTTPS Host·Origin은 허용하고 외부 Origin과 전달 Host 위조는 거부한다."""
        environment = {
            "PUBLIC_ORIGIN": "https://face.example.com",
            "PORT": "8767",
            "DEFAULT_MODEL": "efficientnet_b0",
            "ENABLED_MODELS": "densenet121,efficientnet_b0",
        }
        with patch.dict(os.environ, environment, clear=False):
            public_app = create_app(self.registry)
        public_client = TestClient(public_app, base_url="https://face.example.com")

        accepted = public_client.get("/api/health", headers={"Origin": "https://face.example.com"})
        foreign_origin = public_client.get("/api/health", headers={"Origin": "https://evil.example"})
        spoofed_host = TestClient(public_app, base_url="https://evil.example").get(
            "/api/health",
            headers={"Origin": "https://face.example.com", "X-Forwarded-Host": "face.example.com"},
        )
        docs = public_client.get("/docs", headers={"Origin": "https://face.example.com"})

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(foreign_origin.status_code, 403)
        self.assertEqual(foreign_origin.json(), {"error": "허용되지 않은 요청 출처입니다."})
        self.assertEqual(spoofed_host.status_code, 403)
        self.assertEqual(spoofed_host.json(), {"error": "허용되지 않은 접속 주소입니다. PUBLIC_ORIGIN 설정을 확인해주세요."})
        self.assertEqual(docs.status_code, 404)

    def test_error_shapes_size_limit_and_slot_release(self):
        """404·413·예측 실패 형식과 예외 후 동시 처리 슬롯 반환을 확인한다."""
        missing = self.client.get("/not-found")
        too_large = self.post("/api/crop", data=b"x" * (MAX_BYTES + 1))
        self.registry.fail_next_prediction = True
        failed = self.post("/api/predict?model=efficientnet_b0", token="crop-token")
        recovered = self.post("/api/predict?model=efficientnet_b0", token="crop-token")

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json(), {"error": "없는 경로입니다."})
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(too_large.json(), {"error": "10MB 이하의 사진을 선택해주세요."})
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json(), {"error": "이미지 처리에 실패했습니다. 잠시 후 다시 시도해주세요."})
        self.assertEqual(recovered.status_code, 200)

    def test_upload_errors_and_static_frontend_are_compatible(self):
        """업로드 검증 오류와 기존 프런트엔드 정적 경로를 그대로 제공하는지 확인한다."""
        wrong_type = self.client.post(
            "/api/crop",
            content=b"image",
            headers={"Content-Type": "image/png", "X-Filename": "photo.png", "X-Crop-Token": ""},
        )
        empty = self.post("/api/crop", data=b"")
        index = self.client.get("/")
        asset = self.client.get("/js/model.js")

        self.assertEqual(wrong_type.status_code, 415)
        self.assertEqual(wrong_type.json(), {"error": "이미지 바이너리 요청이 필요합니다."})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.json(), {"error": "빈 이미지입니다."})
        self.assertEqual(index.status_code, 200)
        self.assertIn("text/html", index.headers["Content-Type"])
        self.assertEqual(index.content, (FRONTEND / "index.html").read_bytes())
        self.assertEqual(asset.status_code, 200)
        self.assertEqual(asset.content, (FRONTEND / "js/model.js").read_bytes())

    def test_concurrent_prediction_returns_429(self):
        """같은 ASGI 루프에서 첫 추론 실행 중 두 번째 요청을 기존 429로 거부하는지 확인한다."""
        self.registry.block_predictions = True

        async def scenario():
            """한 ASGI 전송기에서 두 예측 요청을 겹쳐 실행한다."""
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
                headers = {
                    "Content-Type": "application/octet-stream",
                    "X-Filename": "photo.png",
                    "X-Crop-Token": "crop-token",
                }
                first = asyncio.create_task(client.post(
                    "/api/predict?model=efficientnet_b0", content=b"image", headers=headers))
                self.assertTrue(await asyncio.to_thread(self.registry.prediction_entered.wait, 5))
                rejected = await client.post(
                    "/api/predict?model=efficientnet_b0", content=b"image", headers=headers)
                self.registry.release_prediction.set()
                return await first, rejected

        first, rejected = asyncio.run(scenario())
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(rejected.json(), {"error": "다른 사진을 처리 중입니다. 잠시 후 다시 시도해주세요."})
        self.assertEqual(first.status_code, 200)

    def test_asgi_body_length_and_malformed_header_are_rejected(self):
        """ASGI 본문 실제 크기와 선언 길이가 다르거나 헤더가 잘못되면 기존 오류로 거부한다."""
        async def scenario():
            """헤더를 자동 수정하지 않는 ASGI 전송기로 비정상 요청을 보낸다."""
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
                base_headers = {"Content-Type": "application/octet-stream", "X-Filename": "photo.png"}
                mismatch = await client.post(
                    "/api/crop", content=b"image", headers={**base_headers, "Content-Length": "1"})
                malformed = await client.post(
                    "/api/crop", content=b"image", headers={**base_headers, "Content-Length": "invalid"})
                oversized = await client.post(
                    "/api/crop", content=b"x" * (MAX_BYTES + 1),
                    headers={**base_headers, "Content-Length": "1"})
                return mismatch, malformed, oversized

        mismatch, malformed, oversized = asyncio.run(scenario())
        self.assertEqual(mismatch.status_code, 400)
        self.assertEqual(mismatch.json(), {"error": "잘못된 요청입니다."})
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.json(), {"error": "잘못된 요청입니다."})
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.json(), {"error": "10MB 이하의 사진을 선택해주세요."})

    def test_upload_timeout_returns_408_without_waiting(self):
        """30초 업로드 제한이 만료되면 기존 408 오류를 반환하고 슬롯을 해제하는지 확인한다."""
        with patch("backend.controllers.image_controller.asyncio.timeout", return_value=ImmediateTimeout()):
            timed_out = self.post("/api/crop")
        recovered = self.post("/api/crop")

        self.assertEqual(timed_out.status_code, 408)
        self.assertEqual(timed_out.json(), {"error": "사진 전송 시간이 초과되었습니다."})
        self.assertEqual(recovered.status_code, 200)

    def test_production_service_timeout_and_repeated_model_query_match_flask(self):
        """운영 서비스 TimeoutError는 500으로 처리하고 중복 모델 쿼리의 첫 값을 사용한다."""
        with patch.object(self.registry, "predict_crop", side_effect=TimeoutError("model timeout")):
            timed_out = self.post("/api/predict?model=efficientnet_b0", token="crop-token")
        repeated = self.post("/api/predict?model=&model=efficientnet_b0", token="crop-token")

        self.assertEqual(timed_out.status_code, 500)
        self.assertEqual(timed_out.json(), {"error": "이미지 처리에 실패했습니다. 잠시 후 다시 시도해주세요."})
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["model_id"], "")

    def test_swagger_is_disabled_in_production(self):
        """운영 환경에서 OpenAPI 명세와 문서 화면을 노출하지 않는지 확인한다."""
        for path in ("/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), {"error": "없는 경로입니다."})


class LocalFastAPIContractTests(FastAPIContractCase):
    """기존 로컬 실행 모드를 FastAPI로 통합한 계약을 검증한다."""

    base_url = "http://localhost:8766"

    def setUp(self):
        """DenseNet 기본값과 개발 문서를 사용하는 로컬 앱을 만든다."""
        self.registry = FakeRegistry()
        self.app = create_app(self.registry, local=True, port=8766)
        self.client = TestClient(self.app, base_url=self.base_url)

    def test_health_crop_and_default_prediction_model(self):
        """로컬 상태·Crop·예측 경로와 DenseNet 기본 모델을 유지하는지 확인한다."""
        health = self.client.get("/api/health")
        crop = self.post("/api/crop")
        prediction = self.post("/api/predict", crop.content, crop.headers["X-Crop-Token"])

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertNotIn("default_model", health.json())
        self.assertEqual(crop.headers["X-Face-Count"], "1")
        self.assertEqual(prediction.json()["model_id"], "densenet121")

    def test_local_access_policy_and_headers(self):
        """로컬 모드가 현재 포트의 localhost만 허용하고 기존 헤더를 유지하는지 확인한다."""
        bad_host = TestClient(self.app, base_url="http://example.test").get("/api/health")
        bad_origin = self.client.get("/api/health", headers={"Origin": "https://example.test"})
        response = self.client.get("/api/health")

        self.assertEqual(bad_host.status_code, 403)
        self.assertEqual(bad_host.json(), {"error": "localhost에서 접속해주세요."})
        self.assertEqual(bad_origin.status_code, 403)
        self.assertEqual(bad_origin.json(), {"error": "localhost에서 접속해주세요."})
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_local_empty_upload_preserves_413(self):
        """기존 로컬 서버의 빈 본문 응답 상태와 메시지를 유지하는지 확인한다."""
        response = self.post("/api/crop", data=b"")
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"error": "10MB 이하의 사진을 선택해주세요."})

    def test_local_service_timeout_and_repeated_model_query_preserve_handler_rules(self):
        """로컬 서비스 TimeoutError는 408로 처리하고 중복 쿼리의 첫 비어 있지 않은 모델을 쓴다."""
        with patch.object(self.registry, "predict_crop", side_effect=TimeoutError("model timeout")):
            timed_out = self.post("/api/predict?model=densenet121", token="crop-token")
        repeated = self.post("/api/predict?model=&model=efficientnet_b0", token="crop-token")

        self.assertEqual(timed_out.status_code, 408)
        self.assertEqual(timed_out.json(), {"error": "사진 전송 시간이 초과되었습니다."})
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["model_id"], "efficientnet_b0")

    def test_swagger_is_available_only_for_local_development(self):
        """로컬 모드에서만 Swagger와 OpenAPI가 열리고 문서용 CSP가 적용되는지 확인한다."""
        docs = self.client.get("/docs")
        schema = self.client.get("/openapi.json")
        redoc = self.client.get("/redoc")

        self.assertEqual(docs.status_code, 200)
        self.assertIn('id="swagger-ui"', docs.text)
        self.assertIn("cdn.jsdelivr.net", docs.headers["Content-Security-Policy"])
        self.assertEqual(schema.status_code, 200)
        self.assertEqual(schema.json()["openapi"].split(".")[0], "3")
        self.assertEqual(redoc.status_code, 404)


class ServiceContractTests(unittest.TestCase):
    """FastAPI와 무관하게 유지되는 입력 검증과 서비스 조정 책임을 검증한다."""

    def test_upload_validation_preserves_status_codes(self):
        """확장자·전송 형식·본문 크기 오류의 기존 상태 코드를 확인한다."""
        invalid = (
            (("photo.txt", "application/octet-stream", 1), 400),
            (("photo.png", "image/png", 1), 415),
            (("photo.png", "application/octet-stream", 0), 400),
        )
        for arguments, status in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(CropError) as raised:
                    validate_upload(*arguments)
                self.assertEqual(raised.exception.status, status)

    def test_crop_token_is_bound_to_bytes_and_expiry(self):
        """Crop 토큰이 같은 바이트에만 30분 동안 유효한지 확인한다."""
        service = CropService.__new__(CropService)
        service.secret = b"test-secret"
        data = b"cropped-image"
        stamp = str(int(time.time()))
        digest = hmac.new(service.secret, stamp.encode() + b":" + data, hashlib.sha256).hexdigest()
        old_stamp = str(int(stamp) - 1801)
        old_digest = hmac.new(service.secret, old_stamp.encode() + b":" + data, hashlib.sha256).hexdigest()

        service.verify(data, stamp + "." + digest)
        with self.assertRaises(CropError) as changed:
            service.verify(data + b"changed", stamp + "." + digest)
        with self.assertRaises(CropError) as expired:
            service.verify(data, old_stamp + "." + old_digest)
        self.assertEqual(changed.exception.status, 409)
        self.assertEqual(expired.exception.status, 409)

    @patch("backend.services.model_service.CropService")
    @patch("backend.services.model_service.FaceScorer")
    def test_model_registry_selects_model_after_crop_verification(self, scorer_type, crop_type):
        """Registry가 요청 모델을 선택하고 Crop 검증 후에만 추론하는지 확인한다."""
        scorer = scorer_type.return_value
        scorer.predict.return_value = {"score": 3.5}
        registry = ModelRegistry(model_ids=("efficientnet_b0",))
        result = registry.predict_crop(b"crop", "token", "efficientnet_b0")

        crop_type.return_value.verify.assert_called_once_with(b"crop", "token")
        scorer_type.assert_called_once_with(None, "efficientnet_b0")
        scorer.predict.assert_called_once_with(b"crop")
        self.assertEqual(result, {"score": 3.5})
        scorer.predict.reset_mock()
        crop_type.return_value.verify.side_effect = CropError("invalid token", 409)
        with self.assertRaises(CropError):
            registry.predict_crop(b"crop", "invalid", "efficientnet_b0")
        scorer.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
