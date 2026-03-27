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

    def initialize_distortion_from_points(self, pts2d, pts3d):
        """Estimate FOV parameter w from 2D-3D correspondences."""
        with torch.no_grad():
            valid = pts3d[:, 2] > self.EPSILON
            if valid.sum() < 3:
                return
            uv = pts3d[valid, :2] / pts3d[valid, 2:3]
            r_undist = uv.norm(dim=-1)

            uv_pixel = (pts2d[valid] - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
            r_dist = uv_pixel.norm(dim=-1)

            mask = (r_undist > 1e-6) & (r_dist > 1e-6)
            if mask.sum() < 3:
                return

            # r_dist = atan(2*r_undist*tan(w/2)) / w
            # Grid search over w to minimize residual
            best_w, best_err = 0.5, float('inf')
            for w_try in torch.linspace(0.1, 2.0, 40):
                w = w_try.item()
                r_pred = torch.atan(2 * r_undist[mask] * torch.tan(torch.tensor(w / 2))) / w
                err = (r_pred - r_dist[mask]).pow(2).sum().item()
                if err < best_err:
                    best_w, best_err = w, err
            self[4] = best_w

    def _distortion(self, uv):
        r = torch.norm(uv, dim=-1)
        num = torch.atan(2 * r * torch.tan(self[4] / 2))

        return uv * (num / (r + 1e-8) / self[4])[..., None]

    def _undistortion(self, uv):
        r = torch.norm(uv, dim=-1)
        num = torch.tan(r * self[4]) / (r+1e-8) / 2
        return uv * (num / torch.tan(self[4] / 2))[..., None]
