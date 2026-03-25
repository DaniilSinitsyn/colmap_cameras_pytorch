"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import FOV
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(5)
    x[:2] = img_size * 0.6
    x[2:4] = img_size / 2
    x[4] = extra
    return FOV(x, img_size)

class TestFov(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, 1e-4))
        self.models.append(get_model(img_size, 1e-2))
        self.models.append(get_model(img_size, 0.4)  )

        self.model1 = get_model(img_size, 1e-2)
        self.model2 = get_model(img_size, 1e-4)
        self.model2._data.data[:2] *= 0.95
        self.iters = 10
if __name__ == '__main__':
    unittest.main()
