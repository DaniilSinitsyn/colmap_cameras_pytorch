"""
2026 Daniil Sinitsyn

Iterative undistortion — differentiable through Newton iterations.
"""
import torch


def iterative_undistortion(params, pts2d, cam, max_iters):
    """Invert distortion via Newton's method. Differentiable w.r.t. params and pts2d.
    cam must have _distortion() and _d_distortion_d_pts2d() methods."""
    pts = pts2d.clone()
    for _ in range(max_iters):
        f = cam._distortion(pts) - pts2d
        if f.detach().abs().max() < 1e-10:
            break
        J = cam._d_distortion_d_pts2d(pts)
        pts = pts - (torch.linalg.pinv(J) @ f.unsqueeze(-1)).squeeze(-1)
    return pts


class IterativeUndistortion:
    """Backward-compatible wrapper."""
    @staticmethod
    def apply(params, pts2d, cam, max_iters):
        return iterative_undistortion(params, pts2d, cam, max_iters)
