"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch
from ..utils.newton_root_1d import NewtonRoot1D

class OpenCVFisheye(BaseModel):
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

        theta = NewtonRoot1D.apply(r, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)
        
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
