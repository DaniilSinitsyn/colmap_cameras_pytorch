"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch


class OpenCVFisheye(PerspectiveCamera):
    """
    Basically it is the same as simple_radial_fisheye.py
    """
    model_name = 'OPENCV_FISHEYE'
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
        return OpenCVFisheye(x, image_shape)

    def map(self, points3d):
        r = torch.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(r, points3d[:, 2])
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        thetad = theta * (1 + self[4] * theta2 + self[5] * theta4 + self[6] * theta6 + self[7] * theta8)
        
        uv = torch.zeros_like(points3d[:, :2])
        mask = (r < self.EPSILON) | (theta < self.EPSILON)
        uv[mask] = points3d[:, :2][mask]
        uv[~mask] = points3d[:, :2][~mask] * thetad[:, None][~mask] / r[:, None][~mask]

        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), torch.ones_like(r, dtype=torch.bool)

    def map_with_jac(self, points3d):
        # params: [fx, fy, cx, cy, k1, k2, k3, k4]
        N = points3d.shape[0]
        fx = self[0]
        fy = self[1]
        k1 = self[4]
        k2 = self[5]
        k3 = self[6]
        k4 = self[7]

        r = torch.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(r, points3d[:, 2])
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

        uv = torch.zeros_like(points3d[:, :2])
        mask = (r < self.EPSILON) | (theta < self.EPSILON)
        uv[mask] = points3d[:, :2][mask]
        uv[~mask] = points3d[:, :2][~mask] * thetad[:, None][~mask] / r[:, None][~mask]

        pts2d = uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2)
        valid = torch.ones(N, dtype=torch.bool, device=points3d.device)

        J = torch.zeros(N, 2, 8, device=points3d.device, dtype=points3d.dtype)
        # d/dfx = [uv_x, 0]
        J[:, 0, 0] = uv[:, 0]
        # d/dfy = [0, uv_y]
        J[:, 1, 1] = uv[:, 1]
        # d/dcx = [1, 0]
        J[:, 0, 2] = 1.0
        # d/dcy = [0, 1]
        J[:, 1, 3] = 1.0

        # For distortion params: d(result)/dk_i = xy/r * d(thetad)/dk_i * [fx, fy]
        xy_over_r = torch.zeros_like(points3d[:, :2])
        xy_over_r[~mask] = points3d[:, :2][~mask] / r[~mask, None]

        # d(thetad)/dk1 = theta^3, dk2 = theta^5, dk3 = theta^7, dk4 = theta^9
        theta3 = theta2 * theta
        theta5 = theta4 * theta
        theta7 = theta6 * theta
        theta9 = theta8 * theta

        # d/dk1
        J[:, 0, 4] = xy_over_r[:, 0] * theta3 * fx
        J[:, 1, 4] = xy_over_r[:, 1] * theta3 * fy
        # d/dk2
        J[:, 0, 5] = xy_over_r[:, 0] * theta5 * fx
        J[:, 1, 5] = xy_over_r[:, 1] * theta5 * fy
        # d/dk3
        J[:, 0, 6] = xy_over_r[:, 0] * theta7 * fx
        J[:, 1, 6] = xy_over_r[:, 1] * theta7 * fy
        # d/dk4
        J[:, 0, 7] = xy_over_r[:, 0] * theta9 * fx
        J[:, 1, 7] = xy_over_r[:, 1] * theta9 * fy

        return pts2d, valid, J

    def unmap(self, points2d):
        uv = (points2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)

        r = torch.norm(uv, dim=-1)

        polynomials = torch.zeros(r.shape[0], 10).to(r)
        polynomials[:, 9] = self[7]
        polynomials[:, 7] = self[6]
        polynomials[:, 5] = self[5]
        polynomials[:, 3] = self[4]
        polynomials[:, 1] = 1
        polynomials[:, 0] = -r

        theta = self.root_finder(r, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)
        
        mask = (r > self.EPSILON) & (torch.tan(theta) > self.EPSILON)
        z = torch.ones_like(r)
        z[mask] = r[mask] / torch.tan(theta[mask])

        return torch.cat((uv, z[...,None]), dim=-1)

    def initialize_distortion_from_points(self, pts2d, pts3d):
        r = torch.norm(pts3d[:, :2], dim=-1)
        theta = torch.atan2(r, pts3d[:, 2])
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        pts2d = (pts2d - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)
        pts2d = torch.linalg.norm(pts2d, dim=-1)
        
        mask = theta > self.EPSILON
        b = pts2d[mask]/theta[mask] - 1
        A = torch.stack([theta2[mask], theta4[mask], theta6[mask], theta8[mask]], dim=-1)
        x = torch.linalg.lstsq(A, b, rcond=None).solution
        self[4:] = x
