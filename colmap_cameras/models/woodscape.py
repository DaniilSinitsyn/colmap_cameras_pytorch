"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch
from ..utils.newton_root_1d import NewtonRoot1D

class WoodScape(BaseModel):
    model_name = 'WOODSCAPE'
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 4
    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    def map(self, points3d):
        uv = torch.zeros_like(points3d[:, :2])

        chi = torch.linalg.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(chi, points3d[:, 2])

        rho = self[3] * theta + self[4] * theta ** 2 + self[5] * theta ** 3 + self[6] * theta ** 4

        mask = chi > self.EPSILON
        uv[mask] = rho[mask, None] * points3d[:, :2][mask] / chi[mask, None]
        
        uv[:, 1] *= self[0]
        uv += self[1:3].reshape(1, 2) + self.image_shape.reshape(1, 2) / 2 - 0.5
        
        return uv, mask
    
    def get_center(self):
        return self[1:3].reshape(1, 2) + self.image_shape.reshape(1, 2) / 2 - 0.5

    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2) - self.image_shape.reshape(1, 2) / 2 + 0.5) 
        uv[:, 1] = uv[:, 1] / self[0]

        r = torch.linalg.norm(uv, dim=-1)

        polynomials = torch.zeros(r.shape[0], 5).to(r)
        polynomials[:, 4] = self[6]
        polynomials[:, 3] = self[5]
        polynomials[:, 2] = self[4]
        polynomials[:, 1] = self[3]
        polynomials[:, 0] = -r

        theta = NewtonRoot1D.apply(r, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)
       
        mask = (r > self.EPSILON) & (torch.tan(theta) > self.EPSILON)
        z = torch.ones_like(r)
        z[mask] = r[mask] / torch.tan(theta[mask])

        
        return torch.cat((uv, z[...,None]), dim=-1)
