"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch

class UnifiedCameraModel(PerspectiveCamera):
    """
    Unified Camera Model from Mei's paper
    """
    model_name = 'UNIFIED_CAMERA_MODEL'
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
        x[4] = 0.0
        return UnifiedCameraModel(x, image_shape)

    def map(self, points3d):
        d = torch.linalg.norm(points3d, dim=-1)
        valid = points3d[..., 2] + self[4] * d > self.EPSILON
        uv = points3d[:, :2] / (points3d[..., 2] + self[4] * d)[..., None]

        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def initialize_distortion_from_points(self, pts2d, pts3d):
        """Estimate alpha from 2D-3D correspondences."""
        with torch.no_grad():
            fx, fy = self[0], self[1]
            cx, cy = self[2], self[3]

            # Normalize to unit rays
            pts3d_n = pts3d / pts3d.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # Pixel coords -> normalized image coords
            uv_pixel = torch.stack([(pts2d[:, 0] - cx) / fx,
                                    (pts2d[:, 1] - cy) / fy], dim=-1)
            r_pixel = uv_pixel.norm(dim=-1)

            # uv = xy / (z + alpha * d), with d=1 for unit rays
            # With no distortion: r_pixel = ||xy|| / (z + alpha)
            # => alpha = ||xy|| / r_pixel - z
            xy_norm = torch.sqrt(pts3d_n[:, 0] ** 2 + pts3d_n[:, 1] ** 2).clamp(min=1e-8)
            mask = r_pixel > 1e-6
            if mask.sum() < 3:
                return

            alpha_est = xy_norm[mask] / r_pixel[mask] - pts3d_n[mask, 2]
            alpha = alpha_est.median().item()
            alpha = max(alpha, 0.0)
            self[4] = alpha

    def get_center_resolution_focal(self):
        return self._estimate_center_resolution_focal()

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)

        r2 = uv[:, 0] * uv[:, 0] + uv[:, 1] * uv[:, 1]
        b = (self[4] + (1 + (1 - self[4] * self[4]) * r2).sqrt()) / (1 + r2) 
        
        uv = uv * (b / (b - self[4]))[..., None]
        uv[b - self[4] < self.EPSILON] = 0.0

        return torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)



