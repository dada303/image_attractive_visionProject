"""Production contracts, using actual checkpoints (no mock scores)."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import quote
from local_model import ModelRegistry, MAX_BYTES, PROJECT
from production import create_app

class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ModelRegistry()
        with patch.dict(os.environ, {'PORT': '8767', 'PUBLIC_ORIGIN': 'https://face.example.com', 'DEFAULT_MODEL': 'efficientnet_b0'}):
            cls.app = create_app(cls.registry)
        cls.client = cls.app.test_client()
        cls.original = (PROJECT / 'test_photo_9' / 'F_2.png').read_bytes()
        cls.data, cls.token, _ = cls.registry.crops.prepare(cls.original)

    def predict(self, data=None, filename='한글사진.png', **kwargs):
        return self.client.post('/api/predict', base_url='https://face.example.com',
            data=self.data if data is None else data,
            headers={'X-Filename': quote(filename), 'Origin': 'https://face.example.com', 'X-Crop-Token': self.token},
            content_type='application/octet-stream', **kwargs)

    def test_public_https_and_default_efficientnet(self):
        before = {key: id(value.model) for key, value in self.registry.scorers.items()}
        for _ in range(2):
            response = self.predict()
            self.assertEqual(response.status_code, 200)
            result = response.json
            self.assertEqual(result['model_id'], 'efficientnet_b0')
            self.assertAlmostEqual(result['score'], self.registry.predict(self.data, 'efficientnet_b0')['score'], places=5)
        self.assertEqual(before, {key: id(value.model) for key, value in self.registry.scorers.items()})

    def test_reject_host_origin_extension_bytes_size(self):
        self.assertEqual(self.client.get('/', base_url='https://evil.example').status_code, 403)
        self.assertEqual(self.client.post('/api/predict', base_url='https://face.example.com', headers={'Origin':'https://evil.example'}).status_code, 403)
        self.assertEqual(self.predict(filename='x.exe').status_code, 400)
        self.assertEqual(self.predict(data=b'broken').status_code, 409)
        self.assertEqual(self.predict(data=b'x' * (MAX_BYTES + 1)).status_code, 413)

    def test_only_frontend_is_public(self):
        for path in ['/models/efficientnet_b0/EfficientNetAllImage.pt', '/.env', '/local_model.py', '/images/a.jpg', '/api/../.git/config']:
            self.assertEqual(self.client.get(path, base_url='https://face.example.com').status_code, 404)
        for path in ['/', '/js/app.js']:
            with self.client.get(path, base_url='https://face.example.com') as response:
                self.assertEqual(response.status_code, 200)

    def test_single_enabled_model(self):
        with patch.dict(os.environ, {'PUBLIC_ORIGIN':'', 'DEFAULT_MODEL':'efficientnet_b0', 'ENABLED_MODELS':'efficientnet_b0'}):
            app = create_app(ModelRegistry(model_ids=('efficientnet_b0',)))
        result = app.test_client().get('/api/models', base_url='http://localhost:8767').json
        self.assertEqual([m['id'] for m in result['models']], ['efficientnet_b0'])

    def test_crop_never_predicts_and_raw_cannot_predict(self):
        with patch.object(self.registry, 'predict', wraps=self.registry.predict) as predict:
            response = self.client.post('/api/crop', base_url='https://face.example.com', data=self.original,
                content_type='application/octet-stream', headers={'X-Filename':'camera.png'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'image/png')
            predict.assert_not_called()
            self.assertEqual(self.predict(data=self.original).status_code, 409)
            predict.assert_not_called()

    def test_busy_response(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        original = self.registry.predict
        def slow(*args, **kwargs):
            entered.set()
            release.wait(10)
            return original(*args, **kwargs)
        responses = []
        with patch.object(self.registry, 'predict', side_effect=slow):
            def work():
                with self.app.test_client() as client:
                    responses.append(client.post('/api/predict', base_url='https://face.example.com', data=self.data, content_type='application/octet-stream', headers={'X-Filename':'a.png', 'X-Crop-Token':self.token}).status_code)
            thread = threading.Thread(target=work)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.predict().status_code, 429)
            finally:
                release.set()
                thread.join(15)
        self.assertEqual(responses, [200])

if __name__ == '__main__':
    unittest.main(verbosity=2)
