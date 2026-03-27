"""
2026 Daniil Sinitsyn

Camera wrapper that precomputes map/unmap into lookup tables for fast evaluation.
"""
import math
import torch

from .camera_adapter import CameraAdapter


class LUTCamera(CameraAdapter):
    """Precomputes map and unmap into dense lookup tables.

    - unmap LUT: pixel grid -> rays (bilinear interpolation)
    - map LUT: equirectangular (lat, lon) -> pixels (bilinear interpolation)

    The map LUT uses latitude in [-pi/2, pi/2] and longitude in [-pi, pi],
    centered at the optical axis (lat=0, lon=0 = forward). Covers the full
    sphere with no singularity at the forward direction.

    Usage::

        cam = LUTCamera(inner_camera, pixel_step=1, angle_step=0.5)
        pts2d, valid = cam.map(pts3d)   # equirectangular LUT lookup
        rays = cam.unmap(pts2d)          # pixel LUT lookup
    """

    def __init__(self, inner, pixel_step=1, angle_step=1.0):
        """
        Args:
            inner: camera model to wrap
            pixel_step: pixel grid step for the unmap LUT
            angle_step: degrees per cell in the equirectangular map LUT
        """
        super().__init__(inner)
        self._pixel_step = pixel_step
        self._angle_step = angle_step
        self._build_unmap_lut()
        self._build_map_lut()

    def _build_unmap_lut(self):
        """Pixel -> ray LUT."""
        w, h = [int(x.item()) for x in self.inner.image_shape]
        step = self._pixel_step
        u = torch.arange(0, w, step, device=self.inner.device, dtype=torch.float32)
        v = torch.arange(0, h, step, device=self.inner.device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)

        with torch.no_grad():
            rays = self.inner.unmap(pts2d)
            rays[torch.isnan(rays)] = 0

        ws, hs = len(u), len(v)
        self.register_buffer('_unmap_lut', rays.reshape(hs, ws, 3).permute(2, 0, 1))

    def _build_map_lut(self):
        """Equirectangular (lat, lon) -> pixel LUT."""
        step_rad = self._angle_step * math.pi / 180
        n_lat = int(math.pi / step_rad) + 1
        n_lon = int(2 * math.pi / step_rad) + 1

        lat = torch.linspace(-math.pi / 2, math.pi / 2, n_lat, device=self.inner.device)
        lon = torch.linspace(-math.pi, math.pi, n_lon, device=self.inner.device)
        ll_lat, ll_lon = torch.meshgrid(lat, lon, indexing='ij')

        # lat/lon to cartesian: lat=0,lon=0 -> (0,0,1) = forward
        cos_lat = torch.cos(ll_lat.ravel())
        rays = torch.stack([
            cos_lat * torch.sin(ll_lon.ravel()),  # x
            torch.sin(ll_lat.ravel()),              # y
            cos_lat * torch.cos(ll_lon.ravel()),   # z
        ], dim=-1)

        with torch.no_grad():
            pts2d, valid = self.inner.map(rays)
            pts2d[~valid] = -1

        self.register_buffer('_map_lut', pts2d.reshape(n_lat, n_lon, 2).permute(2, 0, 1))
        self.register_buffer('_map_valid', valid.reshape(n_lat, n_lon))

    def map(self, points3d):
        """Equirectangular LUT lookup for projection."""
        N = points3d.shape[0]
        rays_n = points3d / points3d.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Cartesian to lat/lon (lat=0,lon=0 is forward = z-axis)
        lat = torch.asin(rays_n[:, 1].clamp(-1, 1))            # [-pi/2, pi/2]
        lon = torch.atan2(rays_n[:, 0], rays_n[:, 2])           # [-pi, pi]

        # Normalize to [-1, 1] for grid_sample
        grid_x = lon / math.pi                                   # [-1, 1]
        grid_y = lat / (math.pi / 2)                             # [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

        lut = self._map_lut.unsqueeze(0)
        pts2d = torch.nn.functional.grid_sample(
            lut, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        ).reshape(2, N).T

        valid_lut = self._map_valid.unsqueeze(0).unsqueeze(0).float()
        v = torch.nn.functional.grid_sample(
            valid_lut, grid, mode='nearest', padding_mode='zeros', align_corners=True
        ).reshape(N).bool()

        pts2d[~v] = 0
        return pts2d, v

    def unmap(self, points2d):
        """Pixel LUT lookup for backprojection."""
        grid_x = (points2d[:, 0] / self._pixel_step) / (self._unmap_lut.shape[2] - 1) * 2 - 1
        grid_y = (points2d[:, 1] / self._pixel_step) / (self._unmap_lut.shape[1] - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

        lut = self._unmap_lut.unsqueeze(0)
        rays = torch.nn.functional.grid_sample(
            lut, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0).squeeze(1).T

        return rays

    @property
    def model_name(self):
        return f"LUT({self.inner.model_name})"
