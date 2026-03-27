"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch
from ..utils.iterative_undistortion import IterativeUndistortion

class OpenCV(PerspectiveCamera):
    model_name = 'OPENCV'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 4

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(8)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        return OpenCV(x, image_shape)

    def map(self, points3d):
        valid = points3d[:, 2] > 0
        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]
        
        uv[valid] = self._distortion(uv[valid])
        
        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def map_with_jac(self, points3d):
        # params: [fx, fy, cx, cy, k1, k2, p1, p2]
        N = points3d.shape[0]
        P = 8
        valid = points3d[:, 2] > 0

        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]

        uv_undist = uv.clone()
        uv[valid] = self._distortion(uv[valid])

        pts2d = uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2)

        J = torch.zeros(N, 2, P, device=points3d.device, dtype=points3d.dtype)
        # d/dfx = [distorted_x, 0]
        J[:, 0, 0] = uv[:, 0]
        # d/dfy = [0, distorted_y]
        J[:, 1, 1] = uv[:, 1]
        # d/dcx = [1, 0]
        J[:, 0, 2] = 1.0
        # d/dcy = [0, 1]
        J[:, 1, 3] = 1.0

        # d/d[extra_params] = diag([fx, fy]) @ _d_distortion_d_params(uv_undist)
        if valid.any():
            dd_dp = self._d_distortion_d_params(uv_undist[valid])  # (Nv, 2, 4)
            fx = self[0]
            fy = self[1]
            J[valid, 0, 4:] = fx * dd_dp[:, 0, :]
            J[valid, 1, 4:] = fy * dd_dp[:, 1, :]

        return pts2d, valid, J

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
        
        dist_uv = IterativeUndistortion.apply(self[4:], uv, self, self.ROOT_FINDING_MAX_ITERATIONS)

        return torch.cat((dist_uv, torch.ones_like(uv[:, :1])), dim=-1)

    def _distortion(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2
        radial = (self[4] + self[5] * r2) * r2

        tg_u = 2 * self[6] * uv + self[7] * (r2 + 2 * u2)
        tg_v = 2 * self[7] * uv + self[6] * (r2 + 2 * v2)

        new_pts2d = pts2d * (1 + radial[:, None])
        new_pts2d += torch.stack((tg_u, tg_v), dim=-1) 
        
        return new_pts2d
    
    def _d_distortion_d_params(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        uv = pts2d[:, 0] * pts2d[:, 1]
        r2 = u2 + v2

        res = torch.zeros(pts2d.shape[0], 2, 4).to(pts2d)
        res[:, 0, 0] = pts2d[:, 0] * r2
        res[:, 1, 0] = pts2d[:, 1] * r2
        res[:, 0, 1] = res[:, 0, 0] * r2
        res[:, 1, 1] = res[:, 1, 0] * r2

        res[:, 0, 2] = 2 * uv
        res[:, 1, 2] = r2 + 2 * v2
        res[:, 0, 3] = r2 + 2 * u2
        res[:, 1, 3] = res[:, 0, 2]

        return res


    def initialize_distortion_from_points(self, pts2d, pts3d):
        """Estimate k1, k2, p1, p2 from 2D-3D correspondences via linear least squares."""
        with torch.no_grad():
            fx, fy = self[0], self[1]
            cx, cy = self[2], self[3]

            # Undistorted normalized coords from rays (pts3d are unit-length)
            uv_undist = pts3d[:, :2] / pts3d[:, 2:3]

            # Pixel coords -> normalized image coords
            uv_pixel = torch.stack([(pts2d[:, 0] - cx) / fx,
                                    (pts2d[:, 1] - cy) / fy], dim=-1)

            u = uv_undist[:, 0]
            v = uv_undist[:, 1]
            r2 = u ** 2 + v ** 2
            r4 = r2 * r2
            uv_uv = u * v

            # Residual: uv_pixel - uv_undist = uv_undist * (k1*r2 + k2*r4) + [tg_u, tg_v]
            residual = uv_pixel - uv_undist  # (N, 2)

            N = len(pts2d)
            A = torch.zeros(2 * N, 4, device=pts2d.device, dtype=pts2d.dtype)
            b = torch.zeros(2 * N, device=pts2d.device, dtype=pts2d.dtype)

            # u-component
            A[:N, 0] = u * r2
            A[:N, 1] = u * r4
            A[:N, 2] = 2 * uv_uv
            A[:N, 3] = r2 + 2 * u ** 2
            b[:N] = residual[:, 0]

            # v-component
            A[N:, 0] = v * r2
            A[N:, 1] = v * r4
            A[N:, 2] = r2 + 2 * v ** 2
            A[N:, 3] = 2 * uv_uv
            b[N:] = residual[:, 1]

            coeffs = torch.linalg.lstsq(A, b).solution
            self[4] = coeffs[0]  # k1
            self[5] = coeffs[1]  # k2
            self[6] = coeffs[2]  # p1
            self[7] = coeffs[3]  # p2

    def _d_distortion_d_pts2d(self, pts2d):
        u2 = pts2d[:, 0] ** 2
        v2 = pts2d[:, 1] ** 2
        r2 = u2 + v2

        res = torch.eye(2).to(pts2d).unsqueeze(0).repeat(pts2d.shape[0], 1, 1)

        radial = (self[4] + self[5] * r2) * r2
        res *= (1 + radial[:, None])[:, :, None]

        dv = (2 * self[4] + 4 * self[5] * r2) * pts2d[:, 1]
        du = (2 * self[4] + 4 * self[5] * r2) * pts2d[:, 0]

        res[:,0,0] += du * pts2d[:, 0]
        res[:,1,1] += dv * pts2d[:, 1]
        res[:,0,1] += dv * pts2d[:, 0]
        res[:,1,0] += du * pts2d[:, 1]
        
        res[:,0,0] += 2 * self[6] * pts2d[:, 1] + 6 * self[7] * pts2d[:, 0]
        res[:,1,1] += 2 * self[7] * pts2d[:, 0] + 6 * self[6] * pts2d[:, 1]
        res[:,0,1] += 2 * self[6] * pts2d[:, 0] + 2 * self[7] * pts2d[:, 1]
        res[:,1,0] += 2 * self[7] * pts2d[:, 1] + 2 * self[6] * pts2d[:, 0]

        return res

