"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import PolynomialDivisionModel
from .base import TestBase

def get_model(img_size, extra):
    x = torch.zeros(3 + len(extra)).float()
    x[1:3] = img_size / 2
    x[0] = 0.9
    x[3:] = torch.tensor(extra)
    return PolynomialDivisionModel(x, img_size)

class TestPolynomialDivisionModel(TestBase):
    def setUp(self):
        img_size = torch.tensor([100, 100])
        self.models = []
        self.models.append(get_model(img_size, [0,0,0]))
        self.models.append(get_model(img_size, [-8.6947, 27.8308,-59.7256]))

        self.model1 = get_model(img_size, [-5.6947, 25.8308,-49.7256])
        self.model2 = get_model(img_size, [-8.6947, 27.8308,-59.7256])

        self.iters = 10

if __name__ == '__main__':
    unittest.main()
