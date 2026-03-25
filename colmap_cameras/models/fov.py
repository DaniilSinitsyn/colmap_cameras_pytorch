"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch

class FOV(PerspectiveCamera):
    """
    This is a model from 
    Parallel tracking and mapping for small AR workspaces
    """
    model_name = 'FOV'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 1

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)
    
    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(5)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        x[4] = 0.5
        return FOV(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        uv[valid] = self._distortion(uv[valid])

        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
        
        uv_u = self._undistortion(uv)

        return torch.cat((uv_u, torch.ones_like(uv[:, :1])), dim=-1)

    def _distortion(self, uv):
        r = torch.norm(uv, dim=-1)
        num = torch.atan(2 * r * torch.tan(self[4] / 2))

        return uv * (num / (r + 1e-8) / self[4])[..., None]

    def _undistortion(self, uv):
        r = torch.norm(uv, dim=-1)
        num = torch.tan(r * self[4]) / (r+1e-8) / 2
        return uv * (num / torch.tan(self[4] / 2))[..., None]
