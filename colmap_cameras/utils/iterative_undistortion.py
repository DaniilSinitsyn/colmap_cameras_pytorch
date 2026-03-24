"""
2024 Daniil Sinitsyn

Iterative undistortion — differentiable through the Newton iterations.
"""
import torch


def iterative_undistortion(params, pts2d, cam, max_iters):
    """Invert a distortion model via Newton's method.

    Differentiable w.r.t. params and pts2d via autograd through
    the iteration steps.

    Args:
        params: distortion parameters (passed for graph connectivity)
        pts2d: (N, 2) distorted normalized coordinates
        cam: camera model with _distortion() and _d_distortion_d_pts2d()
        max_iters: maximum Newton iterations

    Returns:
        (N, 2) undistorted normalized coordinates
    """
    pts = pts2d.clone()
    for _ in range(max_iters):
        f = cam._distortion(pts) - pts2d
        if f.detach().abs().max() < 1e-10:
            break
        J = cam._d_distortion_d_pts2d(pts)
        J_inv = torch.linalg.pinv(J)
        delta = J_inv @ f.unsqueeze(-1)
        pts = pts - delta.squeeze(-1)
    return pts


# Backward-compatible class name
class IterativeUndistortion:
    @staticmethod
    def apply(params, pts2d, cam, max_iters):
        return iterative_undistortion(params, pts2d, cam, max_iters)
