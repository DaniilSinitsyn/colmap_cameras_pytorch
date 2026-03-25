"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch
from ..utils.iterative_undistortion import IterativeUndistortion

class ThinPrismFisheye(PerspectiveCamera):
    model_name = 'THIN_PRISM_FISHEYE'
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
        return ThinPrismFisheye(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        r = torch.norm(uv, dim=-1)
        theta = torch.atan(r)
        mask = r > self.EPSILON
        uv[mask] *= (theta[mask] / r[mask])[..., None]

        uv[valid] = self._distortion(uv[valid])
        
        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
        
        dist_uv = IterativeUndistortion.apply(self[4:], uv, self, self.ROOT_FINDING_MAX_ITERATIONS)

        theta = torch.norm(dist_uv, dim=-1)
        theta_cos_theta = theta * torch.cos(theta)
        mask  = theta_cos_theta > self.EPSILON

        new_r = torch.ones_like(theta)
        new_r[mask] = (torch.sin(theta[mask]) / theta_cos_theta[mask])


        return torch.cat((dist_uv * new_r[:, None], torch.ones_like(uv[:, :1])), dim=-1)

    def _distortion(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        radial = 1 + (self[4] + (self[5] + (self[8] + self[9] * r2) * r2) * r2) * r2

        tg_u = 2 * self[6] * uv + self[7] * (r2 + 2 * u2) + self[10] * r2
        tg_v = 2 * self[7] * uv + self[6] * (r2 + 2 * v2) + self[11] * r2

        new_pts2d = pts2d * radial[:, None]
        new_pts2d += torch.stack((tg_u, tg_v), dim=-1) 
        
        return new_pts2d
    
    def _d_distortion_d_params(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        res = torch.zeros(pts2d.shape[0], 2, 8).to(pts2d)
        res[:, 0, 0] = pts2d[:, 0] * r2
        res[:, 1, 0] = pts2d[:, 1] * r2
        res[:, 0, 1] = res[:, 0, 0] * r2
        res[:, 1, 1] = res[:, 1, 0] * r2
        res[:, 0, 2] = res[:, 0, 1] * r2
        res[:, 1, 2] = res[:, 1, 1] * r2
        res[:, 0, 3] = res[:, 0, 2] * r2
        res[:, 1, 3] = res[:, 1, 2] * r2

        res[:, 0, 4] = 2 * uv
        res[:, 1, 4] = r2 + 2 * v2
        res[:, 0, 5] = r2 + 2 * u2
        res[:, 1, 5] = res[:, 0, 4]

        res[:, 0, 6] = r2
        res[:, 1, 7] = r2

        return res


    def _d_distortion_d_pts2d(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        r2 = u2 + v2

        radial = 1 + (self[4] + (self[5] + (self[8] + self[9] * r2) * r2) * r2) * r2

        res = torch.eye(2).to(pts2d).unsqueeze(0).repeat(pts2d.shape[0], 1, 1)

        res *= (radial[:, None])[:, :, None]

        dv = (2 * self[4] + 4 * (self[5] + (6 * self[8] + 8 * self[9] * r2) * r2) * r2) * pts2d[:, 1]
        du = (2 * self[4] + 4 * (self[5] + (6 * self[8] + 8 * self[9] * r2) * r2) * r2) * pts2d[:, 0]

        res[:,0,0] += du * pts2d[:, 0]
        res[:,1,1] += dv * pts2d[:, 1]
        res[:,0,1] += dv * pts2d[:, 0]
        res[:,1,0] += du * pts2d[:, 1]
        
        res[:,0,0] += 2 * self[6] * pts2d[:, 1] + 6 * self[7] * pts2d[:, 0] + 2 * self[10] * pts2d[:, 0]
        res[:,1,1] += 2 * self[7] * pts2d[:, 0] + 6 * self[6] * pts2d[:, 1] + 2 * self[11] * pts2d[:, 1]
        res[:,0,1] += 2 * self[6] * pts2d[:, 0] + 2 * self[7] * pts2d[:, 1] + 2 * self[10] * pts2d[:, 1]
        res[:,1,0] += 2 * self[7] * pts2d[:, 1] + 2 * self[6] * pts2d[:, 0] + 2 * self[11] * pts2d[:, 0]

        return res

