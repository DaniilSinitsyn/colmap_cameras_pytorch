"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
from ..utils.companion_matrix_root_1d import CompanionMatrixRoot1D
import torch

class PolynomialDivisionModel(BaseModel):
    """
    DivisionModel
    map:
    (u, v) -> (u-cx, v-cy) / scale
    (u, v, f * (1 + p(||u, v||)))
    """
    model_name = "POLYNOMIAL_DIVISION_MODEL"
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = -1

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)
        self.scale = torch.linalg.norm(image_shape.float())

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(7)
        x[0] = (image_shape[0] + image_shape[1]) / 4
        x[1:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[3:] = 0.0
        return PolynomialDivisionModel(x, image_shape)

    def map(self, points3d):
        valid = torch.ones(points3d.shape[0], dtype=torch.bool, device=points3d.device)
        norm = torch.linalg.norm(points3d[:, :2], dim=-1)

        mask = norm > self.EPSILON
        norm[~mask] = 1
        
        alpha = torch.ones_like(norm)
        z = points3d[:, 2][mask] / norm[mask]
        uv = points3d[:, :2] / norm[..., None]
        
        polynomials = torch.zeros(mask.sum(), self.num_extra_params + 2).to(points3d.device)
        polynomials[:, 2:] = self[3:]
        polynomials[:, 1] = -z
        polynomials[:, 0] = self[0]
       
        roots, valid_roots = CompanionMatrixRoot1D.apply(polynomials)
        
        alpha[mask] = roots
        valid[mask] = valid_roots

        return uv * alpha[...,None] * self.scale + self[1:3].reshape(1, 2), valid

    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2)) / self.scale
        r = torch.norm(uv, dim=-1)
        p = torch.zeros_like(r)
        for i in reversed(range(self.num_extra_params)):
            p = p * r + self[3 + i]
        p = p * r * r + self[0]

        points3d = torch.cat([uv, p[...,None]], dim=-1)
        return points3d

