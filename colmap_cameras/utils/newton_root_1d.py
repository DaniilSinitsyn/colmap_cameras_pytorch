"""
2024 Daniil Sinitsyn

Newton method based polynomial solver — differentiable through the iterations.
"""
import torch


def newton_root_1d(r, polynomial, max_iter):
    """Find a root of a polynomial near initial guess r using Newton's method.

    Differentiable w.r.t. polynomial coefficients via autograd through
    the iteration steps.

    Args:
        r: (B,) initial guess
        polynomial: (B, D) coefficients [a_0, a_1, ..., a_{D-1}]
            representing a_0 + a_1*x + a_2*x^2 + ...
        max_iter: maximum number of Newton steps

    Returns:
        (B,) approximate root
    """
    new_r = r.clone()
    for _ in range(max_iter):
        f = torch.zeros_like(new_r)
        df = torch.zeros_like(new_r)
        for p_i in reversed(range(polynomial.shape[1])):
            f = f * new_r + polynomial[:, p_i]
            if p_i > 0:
                df = df * new_r + polynomial[:, p_i] * p_i
        if f.detach().abs().max() < 1e-10:
            break
        new_r = new_r - f / df
    return new_r


# Keep backward-compatible class name as alias
class NewtonRoot1D:
    @staticmethod
    def apply(r, polynomial, max_iter):
        return newton_root_1d(r, polynomial, max_iter)
