"""
2025 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import Equirectangular
from .base import TestBase

class TestEquirectangular(TestBase):
    def setUp(self):
        img_size = torch.tensor([200, 100])
        x = torch.tensor([200.0, 100.0]) - 1
        self.models = []
        self.models.append(Equirectangular(x, img_size))

if __name__ == '__main__':
    unittest.main()
