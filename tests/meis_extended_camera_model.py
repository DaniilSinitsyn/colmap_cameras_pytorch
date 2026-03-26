"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import MeisExtendedCameraModel, MeisCameraModel
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(10)
    x[:2] = img_size * 1.4
    x[2:4] = img_size / 2
    x[4:] = extra
    return MeisExtendedCameraModel(x, img_size)

class TestMeisExtendedCameraModel(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        self.models.append(get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.0, 0.00042, 0.00042])))
        self.models.append(get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.01, 0.00042, 0.00042])))

        self.model1 = get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.01, 0.00042, 0.00042]))
        self.model2 = get_model(img_size, torch.tensor([2.3, 0.015, 1.6, 0.008, 0.00042, 0.00042]))
        self.iters = 30

class TestMeisExtendedK3Zero(unittest.TestCase):
    def test_k3_zero_matches_base_model(self):
        """Extended model with k3=0 should match the base MeisCameraModel"""
        img_size = torch.tensor([100, 100])

        base_x = torch.zeros(9)
        base_x[:2] = img_size * 1.4
        base_x[2:4] = img_size / 2
        base_x[4:] = torch.tensor([2.4, 0.016, 1.65, 0.00042, 0.00042])
        base_model = MeisCameraModel(base_x, img_size)

        ext_model = get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.0, 0.00042, 0.00042]))

        for u in torch.arange(-0.3, 0.3, 0.1):
            for v in torch.arange(-0.3, 0.3, 0.1):
                pts3d = torch.tensor([[u, v, 1.0], [u, v, 1.0], [u, v, 1.0]]).float()
                pts2d_base, valid_base = base_model.map(pts3d)
                pts2d_ext, valid_ext = ext_model.map(pts3d)
                self.assertTrue(torch.allclose(pts2d_base, pts2d_ext, atol=1e-6))

class TestMeisExtendedK3Effect(unittest.TestCase):
    def test_k3_affects_distortion(self):
        """Non-zero k3 should produce different distortion than k3=0"""
        img_size = torch.tensor([100, 100])
        model_no_k3 = get_model(img_size, torch.tensor([0.0, 0.016, 1.65, 0.0, 0.00042, 0.00042]))
        model_with_k3 = get_model(img_size, torch.tensor([0.0, 0.016, 1.65, 0.5, 0.00042, 0.00042]))

        pts2d = torch.tensor([[0.5, 0.5], [0.3, 0.4], [-0.2, 0.3]]).float()
        d_no_k3 = model_no_k3._distortion(pts2d)
        d_with_k3 = model_with_k3._distortion(pts2d)

        self.assertFalse(torch.allclose(d_no_k3, d_with_k3, atol=1e-6))

    def test_colmap_string_roundtrip(self):
        """to_colmap should produce a parseable string with correct param count"""
        img_size = torch.tensor([100, 100])
        model = get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.3, 0.00042, 0.00042]))
        colmap_str = model.to_colmap()
        parts = colmap_str.split()
        self.assertEqual(parts[0], 'MEIS_EXTENDED_CAMERA_MODEL')
        self.assertEqual(len(parts), 1 + 2 + 10)  # name + image_shape + params

if __name__ == '__main__':
    unittest.main()
