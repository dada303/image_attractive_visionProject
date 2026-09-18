"""Regression checks for backend layout changes. Requires local model files."""
from pathlib import Path
import sys
import os
import unittest
from io import BytesIO
from unittest.mock import patch
from functools import partial
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
import json

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB))
from backend.local_model import ModelRegistry
from backend.production import create_app
from backend.serve import Handler
from PIL import Image

class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ModelRegistry()
        cls.original = (WEB.parent / 'test_photo_9/F_2.png').read_bytes()
        with patch.dict(os.environ, {'PUBLIC_ORIGIN':'', 'PORT':'8767', 'DEFAULT_MODEL':'efficientnet_b0', 'ENABLED_MODELS':'densenet121,mobilenetv3,efficientnet_b0'}):
            cls.app = create_app(cls.registry)
        cls.client = cls.app.test_client()

    def post(self, route, data, token=''):
        return self.client.post(route,base_url='http://localhost:8767',data=data,
            headers={'X-Filename':'photo.png','X-Crop-Token':token},content_type='application/octet-stream')

    def test_crop_and_all_models(self):
        with patch.object(self.registry,'predict',wraps=self.registry.predict) as predict:
            response=self.post('/api/crop',self.original)
            self.assertEqual(response.status_code,200)
            predict.assert_not_called()
        for model in self.registry.scorers:
            result=self.post('/api/predict?model='+model,response.data,response.headers['X-Crop-Token'])
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json['model_id'],model)
            self.assertAlmostEqual(result.json['score'],self.registry.predict(response.data,model)['score'],places=5)
        self.assertEqual(len(self.registry.scorers),3)

    def test_validation(self):
        out=BytesIO();Image.new('RGB',(100,100),'white').save(out,format='PNG')
        self.assertEqual(self.post('/api/crop',out.getvalue()).status_code,422)
        self.assertEqual(self.post('/api/crop',b'bad').status_code,400)
        self.assertEqual(self.post('/api/predict',self.original).status_code,409)
        with self.client.get('/',base_url='http://localhost:8767') as response:
            self.assertEqual(response.status_code,200)
            self.assertIn('결과 보기',response.data.decode())
        self.assertEqual(self.client.get('/models/densenet121/best_model_all.pt',base_url='http://localhost:8767').status_code,404)

    def test_local_http(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,scorer=self.registry))
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://localhost:{server.server_port}'
        try:
            with urlopen(base+'/api/health',timeout=15) as r: self.assertTrue(json.load(r)['ready'])
            with urlopen(base+'/',timeout=15) as r: self.assertIn('결과 보기',r.read().decode())
            headers={'Content-Type':'application/octet-stream','X-Filename':'photo.png'}
            with urlopen(Request(base+'/api/crop',data=self.original,headers=headers),timeout=20) as r:
                crop=r.read();headers['X-Crop-Token']=r.headers['X-Crop-Token']
            with urlopen(Request(base+'/api/predict?model=efficientnet_b0',data=crop,headers=headers),timeout=20) as r:
                self.assertEqual(json.load(r)['model_id'],'efficientnet_b0')
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
