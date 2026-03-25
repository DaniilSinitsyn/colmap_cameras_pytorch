"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch


class SimplePinhole(PerspectiveCamera):
    model_name = 'SIMPLE_PINHOLE'
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 0

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(3)
        x[1:3] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[0] = x[1:3].mean()
        return SimplePinhole(x, image_shape)
    
    @staticmethod
    def from_fov(fov, image_shape):
        x = torch.zeros(3)
        x[1:3] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[0] = x[1:3].mean() / torch.tan(torch.tensor(fov) / 2 * torch.pi / 180) / 2
        return SimplePinhole(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        return uv * self[0] + self[1:3].reshape(1, 2), valid

    def unmap(self, points2d):
        points3d = torch.cat(
            ((points2d - self[1:3].reshape(1, 2)) / self[0], torch.ones_like(
                points2d[:, :1])),
            dim=-1)

        return points3d
