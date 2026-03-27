"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera

import torch

class SimpleRadialFisheye(PerspectiveCamera):
    """
    Distortion is a little bit differen as in colmap
    as it can be changed to exclude division by z,
    potentially giving it a chance to see behind the camera.

    colmap: 
    r = sqrt(x^2 / z^2 + y^2 / z^2)
    theta = atan(r)
    u = (x/z) * (theta + k * theta^3) / r

    here: 
    r = sqrt(x^2 + y^2)
    theta = atan2(sqrt(x^2 + y^2), z)
    u = x * (theta + k * theta^3) / r
    
    new_r for map is (theta + k * theta^3)
    => theta = root(-new_r + theta + k * theta^3)
    """
    model_name = 'SIMPLE_RADIAL_FISHEYE'
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 1

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(4)
        x[1:3] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[0] = x[1:3].mean()
        return SimpleRadialFisheye(x, image_shape)

    def map(self, points3d):
        r = torch.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(r, points3d[:, 2])
      
        theta2 = theta * theta
        thetad = theta * (1 + self[3] * theta2)

        uv = torch.zeros_like(points3d[:, :2])
        mask = (r < self.EPSILON) | (theta < self.EPSILON)
        uv[mask] = points3d[:, :2][mask]
        uv[~mask] = points3d[:, :2][~mask] * thetad[:, None][~mask] / r[:, None][~mask]

        return uv * self[0] + self[1:3].reshape(1, 2), torch.ones_like(r, dtype=torch.bool)
    
    def map_with_jac(self, points3d):
        # params: [f, cx, cy, k1]
        N = points3d.shape[0]
        f = self[0]
        k1 = self[3]

        r = torch.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(r, points3d[:, 2])
        theta2 = theta * theta
        theta3 = theta2 * theta
        thetad = theta * (1 + k1 * theta2)

        uv = torch.zeros_like(points3d[:, :2])
        mask = (r < self.EPSILON) | (theta < self.EPSILON)
        uv[mask] = points3d[:, :2][mask]
        uv[~mask] = points3d[:, :2][~mask] * thetad[:, None][~mask] / r[:, None][~mask]

        pts2d = uv * f + self[1:3].reshape(1, 2)
        valid = torch.ones(N, dtype=torch.bool, device=points3d.device)

        J = torch.zeros(N, 2, 4, device=points3d.device, dtype=points3d.dtype)
        # d/df = uv (the distorted normalized coords)
        J[:, 0, 0] = uv[:, 0]
        J[:, 1, 0] = uv[:, 1]
        # d/dcx
        J[:, 0, 1] = 1.0
        # d/dcy
        J[:, 1, 2] = 1.0
        # d/dk1: for non-degenerate points, xy/r * theta^3 * f
        # For degenerate (mask) points, derivative is 0 (uv = xy, independent of k1)
        xy_over_r = torch.zeros_like(points3d[:, :2])
        xy_over_r[~mask] = points3d[:, :2][~mask] / r[~mask, None]
        J[:, 0, 3] = xy_over_r[:, 0] * theta3 * f
        J[:, 1, 3] = xy_over_r[:, 1] * theta3 * f

        return pts2d, valid, J

    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2)) / self[0]

        r = torch.norm(uv, dim=-1)

        polynomials = torch.zeros(r.shape[0], 4).to(r)
        polynomials[:, 3] = self[3]
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

        pts2d = (pts2d - self[1:3].reshape(1, 2)) / self[0]
        pts2d = torch.linalg.norm(pts2d, dim=-1)

        b = pts2d - theta
        A = theta * theta2
        x = torch.linalg.lstsq(A[:, None], b[:, None], rcond=None)[0]
        self[3] = x
