"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import ThinPrismFisheye
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(12)
    x[:2] = img_size * 0.6
    x[2:4] = img_size / 2
    x[4:] = extra
    return ThinPrismFisheye(x, img_size)

class TestThinPrismFisheye(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, torch.tensor([-0.4, 0.2, -1e-3, 1e-2, 0.001, -0.03, 0.02, -0.01])))

        self.model1 = get_model(img_size, torch.tensor([-0.4, 0.2, -1e-3, 1e-2, 0.001, -0.03, 0.02, -0.01]))
        self.model2 = get_model(img_size, torch.tensor([-0.2, 0.3, -3e-3, 2e-2, 0.002, -0.06, 0.03, -0.02]))
        self.model2._data[:2] *= 0.95

        self.iters = 20

if __name__ == '__main__':
    unittest.main()
