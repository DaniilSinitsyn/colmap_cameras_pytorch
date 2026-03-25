"""
2026 Daniil Sinitsyn

Companion-matrix based polynomial solver with jacobians
"""
import torch

def empty_companion(batch: int, degree: int, device, dtype):
    """
    creates `batch` number of empty companion matricies 
    for polynomial of degree `degree`
    """
    companion = torch.cat(
        (torch.zeros(1, degree - 1), torch.eye(degree - 2, degree - 1)), dim=0)
    
    companion = companion.to(device).to(dtype)
    return companion.repeat(batch, 1, 1)

class CompanionMatrixRoot1D(torch.autograd.Function):
    """
    polynomial is supposed to be a torch.Tensor
    with following structure:
    a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n
    [a_0, a_1, a_2, ..., a_n]
    """
    IMAG_EPS = 1e-4

    @staticmethod
    def forward(ctx, polynomial):
        """
        polynomial - (BxDegree) polynomial coefficients
        """
        b, d = polynomial.shape
        companion = empty_companion(b, d, polynomial.device, polynomial.dtype)
        polynomials = polynomial / (polynomial[...,-1].unsqueeze(-1) + 1e-13)
        companion[..., :, -1] = -polynomials[...,:-1]

        eigs = torch.linalg.eigvals(companion)

        mask = eigs.real > 0.0
        mask = mask & (eigs.imag.abs() < CompanionMatrixRoot1D.IMAG_EPS * eigs.real.abs())

        roots = torch.where(mask, eigs.real, torch.inf).min(dim=-1)[0].reshape(-1, 1)
        roots = roots.reshape(-1)
        valid = roots > 0
        ctx.save_for_backward(polynomial, roots, valid)
        return roots, valid


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, _):
        polynomial, roots, valid = ctx.saved_tensors
        grad_polynomial = None
        
        df = torch.zeros_like(roots)
        for p_i in reversed(range(polynomial.shape[1])):
            if p_i > 0:
                df = df * roots + polynomial[:, p_i] * p_i

        if ctx.needs_input_grad[0]:
            r = torch.ones_like(roots)
            grad_polynomial = torch.zeros_like(polynomial)
            for i in range(0, polynomial.shape[1]):
                grad_polynomial[:,i] = -grad_output* r / df
                r = r * roots

        return grad_polynomial, None
