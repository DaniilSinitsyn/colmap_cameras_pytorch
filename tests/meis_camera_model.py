"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import MeisCameraModel
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(9)
    x[:2] = img_size * 1.4
    x[2:4] = img_size / 2
    x[4:] = extra
    return MeisCameraModel(x, img_size)

class TestMeisCameraModel(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])))
        self.models.append(get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.00042, 0.00042])))

        self.model1 = get_model(img_size, torch.tensor([2.4, 0.016, 1.65, 0.00042, 0.00042]))
        self.model2 = get_model(img_size, torch.tensor([2.3, 0.015, 1.6, 0.00042, 0.00042]))
        self.iters = 30

if __name__ == '__main__':
    unittest.main()
