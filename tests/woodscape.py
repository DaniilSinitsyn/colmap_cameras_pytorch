"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
import torch
from colmap_cameras.models import WoodScape
from .base import TestBase

def get_model(img_size, extra, aspect=1.0):
    x = torch.zeros(3 + len(extra))
    x[0] = aspect
    x[1] = 2.0   # cx offset
    x[2] = -3.0  # cy offset
    x[3:] = torch.tensor(extra)
    return WoodScape(x, img_size)

class TestWoodScape(TestBase):
    def setUp(self):
        img_size = torch.tensor([1280, 966])
        self.models = []
        # Realistic WoodScape parameters (k1≈340 maps radians to pixels)
        self.models.append(get_model(img_size, [339.749, -31.988, 48.275, -7.201]))
        self.models.append(get_model(img_size, [300.0, 0.0, 0.0, 0.0]))  # pure equidistant

        self.iters = 20

if __name__ == '__main__':
    unittest.main()
