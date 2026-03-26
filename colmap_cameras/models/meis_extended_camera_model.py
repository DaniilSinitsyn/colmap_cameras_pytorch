"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch
from ..utils.iterative_undistortion import IterativeUndistortion

class MeisExtendedCameraModel(BaseModel):
    """
    Extended Unified Camera Model from Mei's paper with k3 radial distortion.
    Parameters: fx, fy, cx, cy, alpha, k1, k2, k3, p1, p2
    """
    model_name = 'MEIS_EXTENDED_CAMERA_MODEL'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 1 + 5

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(10)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        return MeisExtendedCameraModel(x, image_shape)

    def map(self, points3d):
        d = torch.linalg.norm(points3d, dim=-1)
        valid = d[..., 2] + self[4] * d > self.EPSILON
        uv = points3d[:, :2] / (points3d[..., 2] + self[4] * d)[..., None]

        uv[valid] = self._distortion(uv[valid])

        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def unmap(self, points2d):
        uv0 = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)

        uv = IterativeUndistortion.apply(self[5:], uv0, self, self.ROOT_FINDING_MAX_ITERATIONS)

        r2 = uv[:, 0] * uv[:, 0] + uv[:, 1] * uv[:, 1]
        b = (self[4] + (1 + (1 - self[4] * self[4]) * r2).sqrt()) / (1 + r2)

        uv = uv * (b / (b - self[4]))[..., None]
        uv[b - self[4] < self.EPSILON] = 0.0

        return torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)


    def _distortion(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2
        radial = (self[5] + (self[6] + self[7] * r2) * r2) * r2

        tg_u = 2 * self[8] * uv + self[9] * (r2 + 2 * u2)
        tg_v = 2 * self[9] * uv + self[8] * (r2 + 2 * v2)

        new_pts2d = pts2d * (1 + radial[:, None])
        new_pts2d += torch.stack((tg_u, tg_v), dim=-1)

        return new_pts2d

    def _d_distortion_d_params(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        res = torch.zeros(pts2d.shape[0], 2, 5).to(pts2d)
        res[:, 0, 0] = pts2d[:, 0] * r2
        res[:, 1, 0] = pts2d[:, 1] * r2
        res[:, 0, 1] = res[:, 0, 0] * r2
        res[:, 1, 1] = res[:, 1, 0] * r2
        res[:, 0, 2] = res[:, 0, 1] * r2
        res[:, 1, 2] = res[:, 1, 1] * r2

        res[:, 0, 3] = 2 * uv
        res[:, 1, 3] = r2 + 2 * v2
        res[:, 0, 4] = r2 + 2 * u2
        res[:, 1, 4] = res[:, 0, 3]

        return res


    def _d_distortion_d_pts2d(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        r2 = u2 + v2

        res = torch.eye(2).to(pts2d).unsqueeze(0).repeat(pts2d.shape[0], 1, 1)

        radial = (self[5] + (self[6] + self[7] * r2) * r2) * r2
        res *= (1 + radial[:, None])[:, :, None]

        dv = (2 * self[5] + (4 * self[6] + 6 * self[7] * r2) * r2) * pts2d[:, 1]
        du = (2 * self[5] + (4 * self[6] + 6 * self[7] * r2) * r2) * pts2d[:, 0]

        res[:,0,0] += du * pts2d[:, 0]
        res[:,1,1] += dv * pts2d[:, 1]
        res[:,0,1] += dv * pts2d[:, 0]
        res[:,1,0] += du * pts2d[:, 1]

        res[:,0,0] += 2 * self[8] * pts2d[:, 1] + 6 * self[9] * pts2d[:, 0]
        res[:,1,1] += 2 * self[9] * pts2d[:, 0] + 6 * self[8] * pts2d[:, 1]
        res[:,0,1] += 2 * self[8] * pts2d[:, 0] + 2 * self[9] * pts2d[:, 1]
        res[:,1,0] += 2 * self[9] * pts2d[:, 1] + 2 * self[8] * pts2d[:, 0]

        return res
