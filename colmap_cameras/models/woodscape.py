"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..perspective_camera import PerspectiveCamera
import torch


class WoodScape(PerspectiveCamera):
    model_name = 'WOODSCAPE'
    num_focal_params = 1
    num_pp_params = 2
    num_extra_params = 4
    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(7)
        x[0] = image_shape[1].float() / 2  # f
        x[1:3] = torch.tensor([image_shape[0], image_shape[1]]).float() / 2  # cx, cy
        x[3] = 1.0  # a (linear term)
        return WoodScape(x, image_shape)

    def map(self, points3d):
        uv = torch.zeros_like(points3d[:, :2])

        chi = torch.linalg.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(chi, points3d[:, 2])

        rho = self[3] * theta + self[4] * theta ** 2 + self[5] * theta ** 3 + self[6] * theta ** 4

        mask = chi > self.EPSILON
        uv[mask] = rho[mask, None] * points3d[:, :2][mask] / chi[mask, None]
        # At chi=0 (optical axis), rho=0 so uv=(0,0) is correct without division

        uv[:, 1] *= self[0]
        uv += self[1:3].reshape(1, 2) + self.image_shape.reshape(1, 2) / 2 - 0.5

        valid = torch.ones_like(chi, dtype=torch.bool)
        return uv, valid
    
    def map_with_jac(self, points3d):
        # params: [f, cx, cy, a, b, c, d]
        N = points3d.shape[0]
        f = self[0]
        a = self[3]
        b = self[4]
        c = self[5]
        d = self[6]

        chi = torch.linalg.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(chi, points3d[:, 2])
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2

        rho = a * theta + b * theta2 + c * theta3 + d * theta4

        uv = torch.zeros_like(points3d[:, :2])
        mask = chi > self.EPSILON
        uv[mask] = rho[mask, None] * points3d[:, :2][mask] / chi[mask, None]

        uv[:, 1] *= f
        uv += self[1:3].reshape(1, 2) + self.image_shape.reshape(1, 2) / 2 - 0.5

        valid = torch.ones(N, dtype=torch.bool, device=points3d.device)

        # Jacobian J is (N, 2, 7)
        J = torch.zeros(N, 2, 7, device=points3d.device, dtype=points3d.dtype)

        # xy/chi for non-degenerate points
        xy_over_chi = torch.zeros_like(points3d[:, :2])
        xy_over_chi[mask] = points3d[:, :2][mask] / chi[mask, None]

        # rho * y / chi is the uv_raw_y before f scaling
        uv_raw_y = torch.zeros(N, device=points3d.device, dtype=points3d.dtype)
        uv_raw_y[mask] = rho[mask] * points3d[:, 1][mask] / chi[mask]

        # d/df = [0, rho * y / chi]
        J[:, 1, 0] = uv_raw_y
        # d/dcx = [1, 0]
        J[:, 0, 1] = 1.0
        # d/dcy = [0, 1]
        J[:, 1, 2] = 1.0
        # d/da: d(rho)/da = theta
        J[:, 0, 3] = xy_over_chi[:, 0] * theta
        J[:, 1, 3] = xy_over_chi[:, 1] * theta * f
        # d/db: d(rho)/db = theta^2
        J[:, 0, 4] = xy_over_chi[:, 0] * theta2
        J[:, 1, 4] = xy_over_chi[:, 1] * theta2 * f
        # d/dc: d(rho)/dc = theta^3
        J[:, 0, 5] = xy_over_chi[:, 0] * theta3
        J[:, 1, 5] = xy_over_chi[:, 1] * theta3 * f
        # d/dd: d(rho)/dd = theta^4
        J[:, 0, 6] = xy_over_chi[:, 0] * theta4
        J[:, 1, 6] = xy_over_chi[:, 1] * theta4 * f

        return uv, valid, J

    def initialize_distortion_from_points(self, pts2d, pts3d):
        """Estimate a, b, c, d polynomial coefficients from 2D-3D correspondences."""
        with torch.no_grad():
            f = self[0]
            center = self[1:3].reshape(1, 2) + self.image_shape.reshape(1, 2) / 2 - 0.5

            # Undo the map: pixel -> uv_raw
            uv_raw = pts2d - center
            uv_raw[:, 1] = uv_raw[:, 1] / f

            # rho = ||uv_raw||
            rho = torch.norm(uv_raw, dim=-1)

            # theta from the 3D rays
            chi = torch.norm(pts3d[:, :2], dim=-1)
            theta = torch.atan2(chi, pts3d[:, 2])

            # Fit rho = a*theta + b*theta^2 + c*theta^3 + d*theta^4
            theta2 = theta * theta
            theta3 = theta2 * theta
            theta4 = theta2 * theta2

            A = torch.stack([theta, theta2, theta3, theta4], dim=-1)
            b = rho

            coeffs = torch.linalg.lstsq(A, b).solution
            self[3] = coeffs[0]  # a
            self[4] = coeffs[1]  # b
            self[5] = coeffs[2]  # c
            self[6] = coeffs[3]  # d

    def get_center(self):
        return self[1:3] + self.image_shape / 2 - 0.5

    def unmap(self, points2d):
        uv = (points2d - self[1:3].reshape(1, 2) - self.image_shape.reshape(1, 2) / 2 + 0.5) 
        uv[:, 1] = uv[:, 1] / self[0]

        r = torch.linalg.norm(uv, dim=-1)

        polynomials = torch.zeros(r.shape[0], 5).to(r)
        polynomials[:, 4] = self[6]
        polynomials[:, 3] = self[5]
        polynomials[:, 2] = self[4]
        polynomials[:, 1] = self[3]
        polynomials[:, 0] = -r

        initial_guess = r / (self[3].detach().abs().clamp(min=1.0))
        theta = self.root_finder(initial_guess, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)
       
        mask = (r > self.EPSILON) & (torch.tan(theta) > self.EPSILON)
        z = torch.ones_like(r)
        z[mask] = r[mask] / torch.tan(theta[mask])

        
        return torch.cat((uv, z[...,None]), dim=-1)
