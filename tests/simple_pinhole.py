"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import SimplePinhole
from .base import TestBase

def get_model(img_size):
    x = torch.zeros(3)
    x[1:3] = img_size / 2
    x[0] = x[1:3].mean()
    return SimplePinhole(x, img_size)

class TestSimplePinhole(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size))

        self.model1 = get_model(img_size)
        self.model2 = get_model(img_size)
        self.model2._data.data[0] *= 0.9

        self.iters = 10

if __name__ == '__main__':
    unittest.main()
