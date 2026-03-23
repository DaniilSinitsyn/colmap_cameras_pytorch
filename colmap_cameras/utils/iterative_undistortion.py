"""
2024 Daniil Sinitsyn

Iterative Undistortion based on implicit function theorem
"""
import torch

class IterativeUndistortion(torch.autograd.Function):
    """
    Assumes the existance of 
    _distortion - distorts 2d points based only on `params`
    _distortion_d_params - jacobian of _distortion with respect to `params`
    _distortion_d_pts2d - jacobian of _distortion with respect to `pts2d`
    methods in cam
    """
    @staticmethod
    def forward(ctx, params, pts2d, cam, max_iters):
        pts = pts2d.clone().detach()
        for _ in range(max_iters):
            f = cam._distortion(pts) - pts2d
            if f.abs().max() < 1e-10:
                break
            J = cam._d_distortion_d_pts2d(pts)
            J_inv = torch.linalg.pinv(J)
            delta = J_inv @ f.unsqueeze(-1)
            pts = pts - delta.squeeze(-1)

        ctx.save_for_backward(pts)
        ctx.cam = cam
        return pts

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        cam = ctx.cam
        pts, = ctx.saved_tensors
        grad_params = None
        grad_pts2d = None
        
        if ctx.needs_input_grad[0]:
            J = cam._d_distortion_d_pts2d(pts)
            J_inv = torch.linalg.pinv(J)
            J_param = cam._d_distortion_d_params(pts)

            J = -J_inv @ J_param
            grad_params = grad_output @ J
        if ctx.needs_input_grad[1]:
            J = cam._d_distortion_d_pts2d(pts)
            J_inv = torch.linalg.pinv(J)
            grad_pts2d = grad_output @ J_inv

        return grad_params, grad_pts2d, None, None


