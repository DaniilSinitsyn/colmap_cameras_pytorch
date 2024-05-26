"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch
from ..utils.newton_root_1d import NewtonRoot1D

class SimpleRadial(BaseModel):
    """
    distortion works as follows:
    (u, v) -> (u, v) * (1 + k1 * r^2)
    meaning r -> new_r = r + k1 * r^3
    the inverse mapping is then a minimum positive real root of the equation
    k1 * r^3 + r - new_r = 0
    """
    model_name = 'SIMPLE_RADIAL'
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 1

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(4)
        x[1:3] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[0] = x[1:3].mean()
        return SimpleRadial(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        u2 = uv[:, 0] * uv[:, 0]
        v2 = uv[:, 1] * uv[:, 1]

        radial = 1 + self[3] * (u2 + v2)
        uv *= radial[:, None]
        return uv * self[0] + self[1:3].reshape(1, 2), valid

    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2)) / self[0]
        
        r2 = uv[:, 0] * uv[:, 0] + uv[:, 1] * uv[:, 1]
        r = torch.ones_like(r2)
        mask = r2 > self.EPSILON

        if abs(self[3]) > self.EPSILON and mask.any():
            r[mask] = torch.sqrt(r2[mask])

            polynomials = torch.zeros(r.shape[0], 4).to(r)
            polynomials[:, 3] = self[3]
            polynomials[:, 1] = 1
            polynomials[:, 0] = -r
            
            new_r = NewtonRoot1D.apply(r, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)
            r[mask] = new_r[mask] / r[mask]

        return torch.cat((uv * r[:, None], torch.ones_like(uv[:, :1])), dim=-1)
