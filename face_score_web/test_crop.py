from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps
from face_crop import CropService, CropError

class CropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = CropService()
        cls.data = (Path(__file__).resolve().parents[1] / 'test_photo_9/F_2.png').read_bytes()

    def encode(self, image):
        out=BytesIO(); image.save(out,format='PNG'); return out.getvalue()

    def test_real_single_and_exact_pixels(self):
        crop, meta=self.service.detect_and_crop_face(self.data)
        with Image.open(BytesIO(self.data)) as im:
            expected=ImageOps.exif_transpose(im).convert('RGB').crop(meta['crop_box'])
        self.assertEqual(crop.tobytes(),expected.tobytes())
        self.assertEqual(crop.width,crop.height)
        self.assertGreater(crop.width,224)

    def test_no_face_and_bad_bytes(self):
        for data,code in [(self.encode(Image.new('RGB',(640,480),'white')),422),(b'broken',400)]:
            with self.assertRaises(CropError) as e: self.service.prepare(data)
            self.assertEqual(e.exception.status,code)

    def test_small_face_large_background(self):
        with Image.open(BytesIO(self.data)) as im:
            thumbnail=ImageOps.contain(im.convert('RGB'),(350,350))
        image=Image.new('RGB',(1400,1000),'#888888');image.paste(thumbnail,(700,400))
        crop,meta=self.service.detect_and_crop_face(self.encode(image))
        self.assertLess(crop.width,600)
        self.assertGreater(meta['face_box'][0],600)

    def test_multiple_real_faces_largest_selected(self):
        with Image.open(BytesIO(self.data)) as im:
            large=ImageOps.contain(im.convert('RGB'),(650,650));small=ImageOps.contain(im.convert('RGB'),(300,300))
        image=Image.new('RGB',(1100,700),'#888888');image.paste(large,(0,0));image.paste(small,(750,100))
        crop,meta=self.service.detect_and_crop_face(self.encode(image))
        self.assertGreaterEqual(meta['face_count'],2)
        self.assertLess(meta['face_box'][0],650)

    def test_camera_png_and_signature(self):
        with Image.open(BytesIO(self.data)) as im: camera=self.encode(im.convert('RGB'))
        png,token,_=self.service.prepare(camera)
        self.service.verify(png,token)
        with self.assertRaises(CropError): self.service.verify(camera,token)
        with patch('face_crop.time.time',return_value=int(token.split('.')[0])+1801):
            with self.assertRaises(CropError): self.service.verify(png,token)

if __name__=='__main__': unittest.main(verbosity=2)
