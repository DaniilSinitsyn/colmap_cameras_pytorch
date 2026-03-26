"""
2026 Daniil Sinitsyn

Composite camera: wraps an inner model with atan continuation past the
valid region boundary and provides a monotonicity regularizer.
"""
import math
import torch

from .camera_adapter import CameraAdapter


def _atan_continuation(r, r_b, theta_b, slope_b):
    """f(r) = theta_b + k * atan((r - r_b) * slope_b / k), asymptotes to pi."""
    k = (math.pi - theta_b) / (math.pi / 2)
    k = k.clamp(min=1e-6) if isinstance(k, torch.Tensor) else max(k, 1e-6)
    return theta_b + k * torch.atan((r - r_b) * slope_b / k)


class CompositeCamera(CameraAdapter):
    """Wraps any camera with atan continuation past the valid boundary.

    theta_b and slope_b are computed live from inner.unmap() at the
    boundary pixels, so gradients flow through to inner camera parameters.

    Only r_boundary (pixel radii per azimuth) is stored as a buffer —
    recomputed via update_boundary() from the valid region mask.

    Usage::

        cam = CompositeCamera(inner)
        cam.update_boundary()
        rays = cam.unmap(pixels)
        loss = cam.monotonicity_loss()
    """

    def __init__(self, inner, n_azimuths=72):
        super().__init__(inner)
        self.n_azimuths = n_azimuths
        self.register_buffer('_r_boundary', torch.full((n_azimuths,), 1e6))
        self.register_buffer('_azimuth_angles',
                             torch.linspace(-math.pi, math.pi, n_azimuths + 1)[:-1])
        # Small offset for finite-difference slope estimation (in pixels)
        self._slope_eps = 1.0

    def update_boundary(self, step=2.0):
        """Recompute boundary radii from the valid region mask."""
        from ..utils.valid_region import estimate_valid_region

        mask = estimate_valid_region(self.inner, step=step)
        mask_np = mask.cpu().numpy()
        h_img, w_img = mask_np.shape

        center = self.inner.get_center().detach()
        if center.numel() < 2:
            center = torch.tensor([w_img / 2.0, h_img / 2.0])
        cx, cy = center[0].item(), center[1].item()

        for i, az in enumerate(self._azimuth_angles):
            dx, dy = math.cos(az.item()), math.sin(az.item())
            r_b = float(max(w_img, h_img))
            for r in range(1, int(r_b)):
                px, py = int(cx + r * dx), int(cy + r * dy)
                if px < 0 or px >= w_img or py < 0 or py >= h_img or not mask_np[py, px]:
                    r_b = float(r)
                    break
            # Back off 15% to stay safely inside the valid region
            self._r_boundary[i] = r_b * 0.85

    def _interp_r_boundary(self, azimuth):
        """Interpolate boundary radius for arbitrary azimuths."""
        az = self._azimuth_angles.to(azimuth.device)
        r_b = self._r_boundary.to(azimuth.device)
        n = len(az)
        idx_f = (torch.atan2(torch.sin(azimuth), torch.cos(azimuth)) - az[0]) / (az[1] - az[0])
        i0, i1 = idx_f.long() % n, (idx_f.long() + 1) % n
        t = idx_f - idx_f.floor()
        return r_b[i0] * (1 - t) + r_b[i1] * t

    def _compute_theta_at_radius(self, center, r, azimuth):
        """Compute ray angle at pixel radius r along given azimuth.
        Returns theta (scalar or batched). Differentiable through inner.unmap()."""
        d = torch.stack([torch.cos(azimuth), torch.sin(azimuth)], dim=-1)
        pts2d = center + r.unsqueeze(-1) * d
        rays = self.inner.unmap(pts2d)
        rays_n = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.acos(rays_n[..., 2].clamp(-1, 1))

    def unmap(self, points2d):
        """Unproject with atan continuation past the boundary.
        theta_b and slope_b are computed live — gradients flow to inner params."""
        center = self.inner.get_center()
        if center.numel() < 2:
            w, h = [int(x.item()) for x in self.inner.image_shape]
            center = torch.tensor([w / 2.0, h / 2.0], device=points2d.device)

        delta = points2d - center.detach()
        pixel_r = delta.norm(dim=-1)
        pixel_az = torch.atan2(delta[:, 1], delta[:, 0])

        r_b = self._interp_r_boundary(pixel_az)
        outside = pixel_r > r_b

        rays = self.inner.unmap(points2d)

        if outside.any():
            az_out = pixel_az[outside]
            r_b_out = r_b[outside]

            # Compute theta and slope at boundary — DIFFERENTIABLE through inner
            theta_b = self._compute_theta_at_radius(center, r_b_out, az_out)
            theta_b_eps = self._compute_theta_at_radius(center, r_b_out - self._slope_eps, az_out)
            slope_b = ((theta_b - theta_b_eps) / self._slope_eps).clamp(min=1e-4)

            theta = _atan_continuation(pixel_r[outside], r_b_out, theta_b, slope_b)

            rays = rays.clone()
            rays[outside] = torch.stack([
                torch.sin(theta) * torch.cos(az_out),
                torch.sin(theta) * torch.sin(az_out),
                torch.cos(theta)], dim=-1)

        return rays

    def monotonicity_loss(self, n_samples=256, eps=1e-6):
        """Penalizes dtheta/dr approaching zero near the boundary.

        Samples radial cuts near the boundary and computes how fast the
        ray angle changes with pixel radius. When this slope approaches
        zero, the distortion is close to folding. Differentiable through
        inner camera parameters via autograd.
        """
        device = self.inner.device
        center = self.inner.get_center()
        if center.numel() < 2:
            w, h = [int(x.item()) for x in self.inner.image_shape]
            center = torch.tensor([w / 2.0, h / 2.0], device=device)

        # Sample near the boundary: radius in [0.5*r_b, 0.95*r_b]
        az = torch.rand(n_samples, device=device) * 2 * math.pi - math.pi
        r_b = self._interp_r_boundary(az)
        r = r_b * (0.5 + 0.45 * torch.rand(n_samples, device=device))

        # Compute theta(r) and theta(r - dr) to get slope
        dr = 1.0
        d = torch.stack([torch.cos(az), torch.sin(az)], dim=-1)
        pts_a = center + r.unsqueeze(-1) * d
        pts_b = center + (r - dr).unsqueeze(-1) * d

        rays_a = self.inner.unmap(pts_a)
        rays_b = self.inner.unmap(pts_b)

        ok = ~torch.isnan(rays_a).any(dim=-1) & ~torch.isnan(rays_b).any(dim=-1)
        if not ok.any():
            return torch.tensor(0.0, device=device)

        theta_a = torch.acos((rays_a / rays_a.norm(dim=-1, keepdim=True).clamp(min=1e-8))[:, 2].clamp(-1, 1))
        theta_b = torch.acos((rays_b / rays_b.norm(dim=-1, keepdim=True).clamp(min=1e-8))[:, 2].clamp(-1, 1))
        slope = (theta_a - theta_b) / dr  # dtheta/dr in radians/pixel

        # Penalize small positive slopes (approaching fold)
        slope_valid = slope[ok]
        return (-torch.log(slope_valid.clamp(min=eps))).mean()

    @property
    def model_name(self):
        return f"Composite({self.inner.model_name})"

    def __repr__(self):
        mean_r = self._r_boundary.mean().item()
        return f"CompositeCamera(\n  {self.inner}\n  mean_boundary_r={mean_r:.1f}px\n)"
