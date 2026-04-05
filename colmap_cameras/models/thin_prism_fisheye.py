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
        # Avoid in-place ops for autograd safety
        z_safe = torch.where(valid, points3d[:, 2], torch.ones_like(points3d[:, 2]))
        uv0 = points3d[:, :2] / z_safe[:, None]

        r = torch.norm(uv0, dim=-1)
        theta = torch.atan(r)
        mask = r > self.EPSILON
        scale = torch.where(mask, theta / r.clamp_min(self.EPSILON), torch.ones_like(r))
        uv1 = uv0 * scale[..., None]

        uv2 = self._distortion(uv1)

        return uv2 * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), valid

    def map_with_jac(self, points3d):
        # params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, s1, s2]
        N = points3d.shape[0]
        P = 12
        valid = points3d[:, 2] > 0

        uv = torch.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]

        r = torch.norm(uv, dim=-1)
        theta = torch.atan(r)
        mask = r > self.EPSILON
        uv[mask] *= (theta[mask] / r[mask])[..., None]

        # uv is now the atan-normalized coords (before distortion)
        uv_normalized = uv.clone()
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

        # d/d[extra_params] = diag([fx, fy]) @ _d_distortion_d_params(uv_normalized)
        # _d_distortion_d_params columns: [k1, k2, k3, k4, p1, p2, s1, s2]
        # self[4:12] storage order:       [k1, k2, p1, p2, k3, k4, s1, s2]
        # So we need to reorder columns to match storage.
        if valid.any():
            dd_dp = self._d_distortion_d_params(uv_normalized[valid])  # (Nv, 2, 8)
            fx = self[0]
            fy = self[1]
            # col -> param index mapping
            col_to_param = [4, 5, 8, 9, 6, 7, 10, 11]
            for col, pidx in enumerate(col_to_param):
                J[valid, 0, pidx] = fx * dd_dp[:, 0, col]
                J[valid, 1, pidx] = fy * dd_dp[:, 1, col]

        return pts2d, valid, J

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

        return pts2d * radial[:, None] + torch.stack((tg_u, tg_v), dim=-1)
    
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


    def initialize_distortion_from_points(self, pts2d, pts3d):
        """Estimate k1, k2, p1, p2 from 2D-3D correspondences (k3, k4, s1, s2 set to 0).
        Uses atan-normalized coordinates matching the map() pipeline."""
        with torch.no_grad():
            fx, fy = self[0], self[1]
            cx, cy = self[2], self[3]

            # Undistorted normalized coords from rays (pts3d are unit-length)
            uv = pts3d[:, :2] / pts3d[:, 2:3]

            # Apply atan normalization (same as map())
            r = torch.norm(uv, dim=-1)
            theta = torch.atan(r)
            mask = r > self.EPSILON
            uv_norm = uv.clone()
            uv_norm[mask] *= (theta[mask] / r[mask])[..., None]

            # Pixel coords -> normalized image coords
            uv_pixel = torch.stack([(pts2d[:, 0] - cx) / fx,
                                    (pts2d[:, 1] - cy) / fy], dim=-1)

            u = uv_norm[:, 0]
            v = uv_norm[:, 1]
            r2 = u ** 2 + v ** 2
            r4 = r2 * r2
            uv_uv = u * v

            # Residual: uv_pixel - uv_norm = uv_norm * (k1*r2 + k2*r4) + [tg_u, tg_v]
            residual = uv_pixel - uv_norm  # (N, 2)

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
            self[4] = coeffs[0]   # k1
            self[5] = coeffs[1]   # k2
            self[6] = coeffs[2]   # p1
            self[7] = coeffs[3]   # p2
            self[8] = 0.0         # k3
            self[9] = 0.0         # k4
            self[10] = 0.0        # s1
            self[11] = 0.0        # s2

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

