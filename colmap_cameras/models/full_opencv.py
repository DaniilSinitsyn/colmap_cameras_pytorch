"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch
from ..utils.iterative_undistortion import IterativeUndistortion

class FullOpenCV(PerspectiveCamera):
    model_name = 'FULL_OPENCV'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 8

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(12)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        return FullOpenCV(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        uv[valid] = self._distortion(uv[valid])
        
        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
        
        dist_uv = IterativeUndistortion.apply(self[4:], uv, self, self.ROOT_FINDING_MAX_ITERATIONS)

        return torch.cat((dist_uv, torch.ones_like(uv[:, :1])), dim=-1)

    def _distortion(self, pts2d):
        # COLMAP order: k1, k2, p1, p2, k3, k4, k5, k6
        k1, k2, p1, p2, k3, k4, k5, k6 = [self[4+i] for i in range(8)]

        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        radial = 1 + (k1 + (k2 + k3 * r2) * r2) * r2
        radial /= (1 + (k4 + (k5 + k6 * r2) * r2) * r2)

        tg_u = 2 * p1 * uv + p2 * (r2 + 2 * u2)
        tg_v = 2 * p2 * uv + p1 * (r2 + 2 * v2)

        new_pts2d = pts2d * radial[:, None]
        new_pts2d += torch.stack((tg_u, tg_v), dim=-1)

        return new_pts2d
    
    def _d_distortion_d_params(self, pts2d):
        # COLMAP order: k1, k2, p1, p2, k3, k4, k5, k6
        k1, k2, p1, p2, k3, k4, k5, k6 = [self[4+i] for i in range(8)]

        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        num = 1 + (k1 + (k2 + k3 * r2) * r2) * r2
        denom = (1 + (k4 + (k5 + k6 * r2) * r2) * r2)
        denom_inv = 1.0 / denom
        denom2 = denom_inv * denom_inv

        res = torch.zeros(pts2d.shape[0], 2, 8).to(pts2d)
        # d/dk1 (index 0)
        res[:, 0, 0] = pts2d[:, 0] * r2 * denom_inv
        res[:, 1, 0] = pts2d[:, 1] * r2 * denom_inv
        # d/dk2 (index 1)
        res[:, 0, 1] = res[:, 0, 0] * r2
        res[:, 1, 1] = res[:, 1, 0] * r2
        # d/dp1 (index 2)
        res[:, 0, 2] = 2 * uv
        res[:, 1, 2] = r2 + 2 * v2
        # d/dp2 (index 3)
        res[:, 0, 3] = r2 + 2 * u2
        res[:, 1, 3] = 2 * uv
        # d/dk3 (index 4)
        res[:, 0, 4] = res[:, 0, 0] * r2 * r2
        res[:, 1, 4] = res[:, 1, 0] * r2 * r2
        # d/dk4 (index 5)
        res[:, 0, 5] = -pts2d[:, 0] * num * r2 * denom2
        res[:, 1, 5] = -pts2d[:, 1] * num * r2 * denom2
        # d/dk5 (index 6)
        res[:, 0, 6] = res[:, 0, 5] * r2
        res[:, 1, 6] = res[:, 1, 5] * r2
        # d/dk6 (index 7)
        res[:, 0, 7] = res[:, 0, 6] * r2
        res[:, 1, 7] = res[:, 1, 6] * r2

        return res


    def _d_distortion_d_pts2d(self, pts2d):
        # COLMAP order: k1, k2, p1, p2, k3, k4, k5, k6
        k1, k2, p1, p2, k3, k4, k5, k6 = [self[4+i] for i in range(8)]

        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        r2 = u2 + v2

        num = 1 + (k1 + (k2 + k3 * r2) * r2) * r2
        denom = (1 + (k4 + (k5 + k6 * r2) * r2) * r2)

        res = torch.eye(2).to(pts2d).unsqueeze(0).repeat(pts2d.shape[0], 1, 1)
        res *= (num / denom)[..., None, None]

        dv_t = (2 * k1 + (4 * k2 + 6 * k3 * r2) * r2) * pts2d[:, 1]
        du_t = (2 * k1 + (4 * k2 + 6 * k3 * r2) * r2) * pts2d[:, 0]

        dv_d = (2 * k4 + (4 * k5 + 6 * k6 * r2) * r2) * pts2d[:, 1]
        du_d = (2 * k4 + (4 * k5 + 6 * k6 * r2) * r2) * pts2d[:, 0]

        denom2 = 1.0 / (denom * denom)

        dv = (dv_t * denom - dv_d * num) * denom2
        du = (du_t * denom - du_d * num) * denom2

        res[:,0,0] += pts2d[:, 0] * du
        res[:,1,1] += pts2d[:, 1] * dv
        res[:,0,1] += pts2d[:, 0] * dv
        res[:,1,0] += pts2d[:, 1] * du

        res[:,0,0] += 2 * p1 * pts2d[:, 1] + 6 * p2 * pts2d[:, 0]
        res[:,1,1] += 2 * p2 * pts2d[:, 0] + 6 * p1 * pts2d[:, 1]
        res[:,0,1] += 2 * p1 * pts2d[:, 0] + 2 * p2 * pts2d[:, 1]
        res[:,1,0] += 2 * p2 * pts2d[:, 1] + 2 * p1 * pts2d[:, 0]

        return res
