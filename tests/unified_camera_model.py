"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import UnifiedCameraModel
from .base import TestBase

def get_model(img_size, xi):
    x = torch.zeros(5)
    x[:2] = img_size * 0.6
    x[2:4] = img_size / 2
    x[4] = xi
    return UnifiedCameraModel(x, img_size)

class TestUnifiedCameraModel(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, 0.0))

        self.model1 = get_model(img_size, 1.2)
        self.model2 = get_model(img_size, 1.5)
        self.model2._data.data[:2] *= 0.9
        self.iters = 10

if __name__ == '__main__':
    unittest.main()
