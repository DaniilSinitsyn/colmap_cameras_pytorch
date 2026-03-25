"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import FullOpenCV
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(12)
    x[:2] = img_size * 0.6
    x[2:4] = img_size / 2
    x[4:] = extra
    return FullOpenCV(x, img_size)

class TestFullOpenCV(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, torch.tensor([-0.4, 0.2, -1e-3, 1e-3, 1e-4, 0.04, -0.01, 1e-3])))

        self.model1 = get_model(img_size, torch.tensor([-0.4, 0.2, -1e-3, 1e-3, 1e-4, 0.04, -0.01, 1e-3]))
        self.model2 = get_model(img_size, torch.tensor([-0.4, 0.25, -3e-3, 2e-3, 5e-4, 0.03, -0.02, 1e-3]))
        self.model2._data.data[:2] *= 0.95

        self.iters = 10

if __name__ == '__main__':
    unittest.main()
