"""Integration checks against the real checkpoint and a temporary localhost server."""
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from local_model import ModelRegistry, PROJECT
from serve import Handler

class LocalAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = ModelRegistry()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, scorer=cls.scorer))
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://localhost:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, route, data=None, **headers):
        return urlopen(Request(self.url + route, data=data, headers=headers), timeout=30)

    def test_real_image_matches_direct_inference(self):
        photo = next((PROJECT / 'test_photo_9').glob('*.png'))
        data = photo.read_bytes()
        for model_id in ('densenet121', 'mobilenetv3', 'efficientnet_b0'):
            with self.subTest(model=model_id):
                direct = self.scorer.predict(data, model_id)
                with self.request('/api/predict?model=' + model_id, data) as response:
                    result = json.load(response)
                self.assertEqual(result['model_id'], model_id)
                self.assertAlmostEqual(result['raw_score'], direct['raw_score'], places=6)
                self.assertGreaterEqual(result['score'], 1)
                self.assertLessEqual(result['score'], 5)
                if model_id == 'efficientnet_b0':
                    import math
                    self.assertAlmostEqual(result['score'], 1 + 4 / (1 + math.exp(-result['raw_score'])), places=5)
                print(f"{model_id}: {result['score']:.4f}")

    def test_unknown_model_rejected(self):
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/predict?model=unknown', b'data')
        self.assertEqual(caught.exception.code, 400)

    def test_models_list(self):
        with self.request('/api/models') as response:
            models = json.load(response)['models']
        self.assertEqual(len(models), 3)
        self.assertTrue(all(model['ready'] for model in models))

    def test_invalid_image(self):
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/predict', b'not an image')
        self.assertEqual(caught.exception.code, 400)

    def test_foreign_origin_rejected(self):
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/predict', b'data', Origin='https://example.com')
        self.assertEqual(caught.exception.code, 403)

    def test_oversized_dimensions(self):
        data = BytesIO()
        Image.new('RGB', (5000, 4001)).save(data, format='PNG')
        with self.assertRaisesRegex(ValueError, '화소'):
            self.scorer.predict(data.getvalue())

    def test_health_and_ui(self):
        with self.request('/api/health') as response:
            self.assertTrue(json.load(response)['ready'])
        with self.request('/') as response:
            self.assertIn('점수 확인', response.read().decode('utf-8'))
        with self.request('/js/app.js') as response:
            self.assertIn('predictImage(selectedBlob, modelId)', response.read().decode('utf-8'))

if __name__ == '__main__':
    unittest.main(verbosity=2)
