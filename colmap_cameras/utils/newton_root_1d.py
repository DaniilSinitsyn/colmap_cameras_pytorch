"""
2026 Daniil Sinitsyn

Newton method polynomial root solver — differentiable through iterations.
"""
import torch


def newton_root_1d(r, polynomial, max_iter):
    """Find root of polynomial near initial guess r.
    polynomial: (B, D) coefficients [a_0, a_1, ..., a_{D-1}].
    Differentiable w.r.t. polynomial via autograd."""
    new_r = r.clone()
    for _ in range(max_iter):
        f = torch.zeros_like(new_r)
        df = torch.zeros_like(new_r)
        for i in reversed(range(polynomial.shape[1])):
            f = f * new_r + polynomial[:, i]
            if i > 0:
                df = df * new_r + polynomial[:, i] * i
        if f.detach().abs().max() < 1e-10:
            break
        new_r = new_r - f / df
    return new_r


class NewtonRoot1D:
    """Backward-compatible wrapper."""
    @staticmethod
    def apply(r, polynomial, max_iter):
        return newton_root_1d(r, polynomial, max_iter)
