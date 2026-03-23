"""
2024 Daniil Sinitsyn

Newton method based polynomial solver with jacobians
"""
import torch

class NewtonRoot1D(torch.autograd.Function):
    """
    polynomial is supposed to be a torch.Tensor
    with following structure:
    a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n
    [a_0, a_1, a_2, ..., a_n]
    """
    @staticmethod
    def forward(ctx, r, polynomial, max_iter):
        """
        r -          (B) initial estimation for the root
        polynomial - (BxDegree) polynomial coefficients
        """
        new_r = r.clone().detach()
        for iteration in range(max_iter):
            f = torch.zeros_like(new_r)
            df = torch.zeros_like(new_r)
            for p_i in reversed(range(polynomial.shape[1])):
                f = f * new_r + polynomial[:, p_i]
                if p_i > 0:
                    df = df * new_r + polynomial[:, p_i] * p_i
            if f.abs().max() < 1e-10:
                break
            new_r = new_r - f / df
        
        df = torch.zeros_like(new_r)
        for p_i in reversed(range(polynomial.shape[1])):
            if p_i > 0:
                df = df * new_r + polynomial[:, p_i] * p_i
        
        ctx.save_for_backward(df, new_r, polynomial)
        return new_r

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        df, new_r, polynomial = ctx.saved_tensors
        
        grad_polynomial = None
        
        if ctx.needs_input_grad[1]:
            r = torch.ones_like(new_r)
            grad_polynomial = torch.zeros_like(polynomial)
            for i in range(0, polynomial.shape[1]):
                grad_polynomial[:,i] = -grad_output* r / df
                r = r * new_r

        return None, grad_polynomial, None
