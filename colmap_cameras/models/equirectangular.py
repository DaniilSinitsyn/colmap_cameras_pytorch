"""
2025 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch

class Equirectangular(BaseModel):
    model_name = 'EQUIRECTANGULAR'
    num_focal_params = 2
    num_pp_params = 0
    num_extra_params = 0

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = (image_shape - 1).float()
        return Equirectangular(x, image_shape)

    def map(self, points3d):
        valid = torch.ones_like(points3d[:, 2]).bool()
        pts3d = torch.nn.functional.normalize(points3d, dim=-1)

        phi = torch.asin(pts3d[:, 1])
        theta = torch.atan2(pts3d[:, 0], pts3d[:, 2])

        uv = torch.zeros_like(points3d[:, :2])
        uv[:, 0] = (theta / (2.0 * torch.pi) + 0.5) * self._data[0]
        uv[:, 1] = (phi / torch.pi + 0.5) * self._data[1]
        
        return uv, valid

    def unmap(self, points2d):
        theta = ((points2d[:, 0] / self._data[0]) - 0.5) * (2.0 * torch.pi)
        phi = ((points2d[:, 1] / self._data[1]) - 0.5) * torch.pi
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = torch.cos(phi) * torch.cos(theta)
        points3d = torch.stack((x, y, z), dim=-1)
        return points3d
