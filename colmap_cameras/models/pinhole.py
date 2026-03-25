"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch

class Pinhole(PerspectiveCamera):
    model_name = 'PINHOLE'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 0

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(4)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        return Pinhole(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def unmap(self, points2d):
        points3d = torch.cat(
                ((points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2), torch.ones_like(
                points2d[:, :1])),
            dim=-1)

        return points3d
