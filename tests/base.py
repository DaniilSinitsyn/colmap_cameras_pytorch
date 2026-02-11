"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
from .common import test_pt2d, test_pt3d, test_model_fit, test_model_3dpts_fil

class TestBase(unittest.TestCase):
    def test_map(self):
        for model in self.models:
            test_pt2d(model, self)

    def test_unmap(self):
        for model in self.models:
            test_pt3d(model, self)

    def test_model_fit(self):
        if hasattr(self, 'model1') and hasattr(self, 'model2') and hasattr(self, 'iters'):
            test_model_fit(self.model1, self.model2, self.iters, self)
    
    def test_model_3dpts_fil(self):
        if hasattr(self, 'model1') and hasattr(self, 'model2'):
            test_model_3dpts_fil(self.model1, self.model2, self)
