"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import DivisionModel
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(4).float()
    x[1:3] = img_size / 2
    x[0] = 0.9
    x[3:] = extra
    return DivisionModel(x, img_size)

class TestDivisionModel(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, 0))
        self.models.append(get_model(img_size, -1.0))

        self.model1 = get_model(img_size, -1.0)
        self.model2 = get_model(img_size, -0.15)
        self.model2._data.data[0] *= 0.9

        self.iters = 10

if __name__ == '__main__':
    unittest.main()
