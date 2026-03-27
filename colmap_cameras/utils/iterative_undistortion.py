"""
2026 Daniil Sinitsyn

Iterative undistortion — differentiable through Newton iterations.
"""
import torch


def _inv2x2(J):
    """Analytical inverse of a batch of 2×2 matrices (N, 2, 2). Much faster than pinv/SVD."""
    a, b = J[:, 0, 0], J[:, 0, 1]
    c, d = J[:, 1, 0], J[:, 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / det.clamp(min=1e-12)
    inv = torch.stack([
        torch.stack([ d, -b], dim=-1),
        torch.stack([-c,  a], dim=-1),
    ], dim=-2) * inv_det[:, None, None]
    return inv


def iterative_undistortion(params, pts2d, cam, max_iters):
    """Invert distortion via Newton's method. Differentiable w.r.t. params and pts2d.
    cam must have _distortion() and _d_distortion_d_pts2d() methods."""
    pts = pts2d.clone()
    for _ in range(max_iters):
        f = cam._distortion(pts) - pts2d
        if f.detach().abs().max() < 1e-10:
            break
        J = cam._d_distortion_d_pts2d(pts)
        pts = pts - (_inv2x2(J) @ f.unsqueeze(-1)).squeeze(-1)
    return pts


class IterativeUndistortion:
    """Backward-compatible wrapper."""
    @staticmethod
    def apply(params, pts2d, cam, max_iters):
        return iterative_undistortion(params, pts2d, cam, max_iters)
