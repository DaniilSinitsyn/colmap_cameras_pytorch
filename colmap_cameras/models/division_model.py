"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch

class DivisionModel(BaseModel):
    """
    DivisionModel
    map:
    (u, v) -> (u-cx, v-cy) / scale
    (u, v, f * (1 + lambda * ||u, v||^2))
    """
    model_name = "DIVISION_MODEL"
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 1

    def __init__(self, x, image_shape, scale_from_focal=True):
        super().__init__(x, image_shape)
        if scale_from_focal:
            self.register_buffer('scale', x[0].detach().clone())
            self[0] = 1.0
        else:
            self.register_buffer('scale', torch.linalg.norm(image_shape.float()))

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(4)
        x[0] = (image_shape[0] + image_shape[1]) / 4
        x[1:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[3] = 0.0
        return DivisionModel(x, image_shape)

    def map(self, points3d):

        r2 = points3d[:, 0] ** 2 + points3d[:, 1] ** 2
        z = points3d[:, 2]
        valid = (z*z >= 4 * self[0]**2 * self[3] * r2)

        mask = r2 > self.EPSILON

        alpha = self[0] / z
        if self[3].abs() > self.EPSILON:
            new_alpha = (z[mask] - torch.sqrt(z[mask] * z[mask] - 4 * self[0]**2 * self[3] * r2[mask])) / (2 * self[0] * self[3] * r2[mask])
            alpha[mask] = new_alpha

        uv = self.scale * alpha[...,None] * points3d[:, 0:2]
        return uv + self[1:3].reshape(1, 2), valid


    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2)) / self.scale

        r2 = uv[:, 0] ** 2 + uv[:, 1] ** 2
        z = self[0] * (1 + self[3] * r2)
        points3d = torch.cat([uv, z[...,None]], dim=-1)
        return points3d

