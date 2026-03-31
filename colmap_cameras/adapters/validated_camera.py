"""
2026 Daniil Sinitsyn

Camera wrapper that filters points through a precomputed spherical validity map.
"""
import math
import torch

from .camera_adapter import CameraAdapter
from ..utils.valid_region import estimate_valid_region


class ValidatedCamera(CameraAdapter):
    """Wraps any camera and masks out points outside the valid region.

    Precomputes a validity lookup in spherical coordinates (theta, phi).
    Both map() and unmap() return (result, valid) tuples with zeros for invalid.

    Usage::

        cam = ValidatedCamera(inner_camera)
        pts2d, valid = cam.map(pts3d)
        rays, valid = cam.unmap(pts2d)
    """

    def __init__(self, inner, step=2.0, angle_step=1.0, auto_update=True):
        super().__init__(inner)
        self._step = step
        self._angle_step = angle_step
        n_theta = int(180 / angle_step) + 1
        n_phi = int(360 / angle_step) + 1
        self.register_buffer('_sphere_valid', torch.ones(n_theta, n_phi, dtype=torch.bool, device=inner.device))
        if auto_update:
            self.update(step)

    def update(self, step=None):
        """Recompute spherical validity map from pixel valid mask."""
        pixel_mask = estimate_valid_region(self.inner, step=step or self._step)
        self._build_sphere_map(pixel_mask)

    def _build_sphere_map(self, pixel_mask):
        device = self.inner.device
        h, w = pixel_mask.shape
        u = torch.arange(0, w, device=device, dtype=torch.float32)
        v = torch.arange(0, h, device=device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)
        valid_pts = pts2d[pixel_mask.T.ravel()]

        self._sphere_valid.fill_(False)
        if len(valid_pts) == 0:
            return

        with torch.no_grad():
            rays = self.inner.unmap(valid_pts)
        ok = ~torch.isnan(rays).any(dim=-1)
        rays = rays[ok]
        rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        theta = torch.acos(rays[:, 2].clamp(-1, 1)) * 180 / math.pi
        phi = torch.atan2(rays[:, 1], rays[:, 0]) * 180 / math.pi

        ti = (theta / self._angle_step).long().clamp(0, self._sphere_valid.shape[0] - 1)
        pi = ((phi + 180) / self._angle_step).long().clamp(0, self._sphere_valid.shape[1] - 1)
        self._sphere_valid[ti, pi] = True

        # Dilate by 1 cell to close discretization gaps at bucket boundaries
        padded = torch.nn.functional.pad(self._sphere_valid.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='replicate')
        dilated = torch.nn.functional.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
        self._sphere_valid = dilated.squeeze().bool()

    def _rays_valid(self, rays):
        with torch.no_grad():
            r = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            theta = torch.acos(r[:, 2].clamp(-1, 1)) * 180 / math.pi
            phi = torch.atan2(r[:, 1], r[:, 0]) * 180 / math.pi
            ti = (theta / self._angle_step).long().clamp(0, self._sphere_valid.shape[0] - 1)
            pi = ((phi + 180) / self._angle_step).long().clamp(0, self._sphere_valid.shape[1] - 1)
            return self._sphere_valid[ti, pi]

    def map(self, points3d):
        pts2d, valid = self.inner.map(points3d)
        ray_ok = self._rays_valid(points3d)
        valid = valid & ray_ok
        pts2d[~valid] = 0
        return pts2d, valid

    def unmap(self, points2d):
        """Returns (rays, valid). Invalid rays are zero vectors."""
        rays = self.inner.unmap(points2d)
        ok = ~torch.isnan(rays).any(dim=-1)
        ray_ok = torch.zeros(len(rays), dtype=torch.bool, device=rays.device)
        if ok.any():
            ray_ok[ok] = self._rays_valid(rays[ok])
        rays[~ray_ok] = 0
        return rays, ray_ok

    @property
    def valid_mask(self):
        return self._sphere_valid

    @property
    def model_name(self):
        return f"Validated({self.inner.model_name})"
