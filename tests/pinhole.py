"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import Pinhole
from .base import TestBase

def get_model(img_size):
    x = torch.zeros(4)
    x[:2] = img_size * 0.6
    x[2:4] = img_size / 2
    return Pinhole(x, img_size)

class TestPinhole(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size))

        self.model1 = get_model(img_size)
        self.model2 = get_model(img_size)
        self.model2._data.data[:2] *= 0.7
        self.iters = 20

if __name__ == '__main__':
    unittest.main()
