"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch


class BaseCamera(torch.nn.Module):
    """Base camera model. Parameters stored as nn.Parameter with per-element gradient mask."""

    _image_shape: torch.Tensor
    _data: torch.nn.Parameter
    ROOT_FINDING_MAX_ITERATIONS = 20
    ROOT_FINDING_METHOD = 'companion'  # 'newton' or 'companion'
    EPSILON = 1e-6

    def root_finder(self, initial_guess, polynomial, max_iter):
        """Polynomial root finder. Set ROOT_FINDING_METHOD to switch solver."""
        if self.ROOT_FINDING_METHOD == 'companion':
            from .utils.companion_matrix_root_1d import CompanionMatrixRoot1D
            roots, _ = CompanionMatrixRoot1D.apply(polynomial)
            return roots
        from .utils.newton_root_1d import newton_root_1d
        return newton_root_1d(initial_guess, polynomial, max_iter)

    def __init__(self, data, image_shape):
        super().__init__()

        if self.num_extra_params == -1:
            if data.shape[0] < self.num_focal_params + self.num_pp_params + 1:
                raise ValueError(f"Expected at least {self.num_focal_params + self.num_pp_params + 1} params, got {data.shape[0]}")
            self.num_extra_params = data.shape[0] - self.num_focal_params - self.num_pp_params

        expected = self.num_focal_params + self.num_pp_params + self.num_extra_params
        if data.shape[0] != expected:
            raise ValueError(f"Expected {expected} params, got {data.shape[0]}")

        self._data = data if isinstance(data, torch.nn.Parameter) else torch.nn.Parameter(data, requires_grad=False)
        self.register_buffer('_image_shape', image_shape)
        self.register_buffer('_grad_mask', torch.ones(data.shape[0], dtype=data.dtype, device=data.device))

    # ----- Fix / unfix --------------------------------------------------------

    def _mask_gradient(self, grad):
        return grad * self._grad_mask

    _PARAM_GROUPS = {'focal': lambda s: range(0, s.num_focal_params),
                     'center': lambda s: range(s.num_focal_params, s.num_focal_params + s.num_pp_params),
                     'extra': lambda s: range(s.num_focal_params + s.num_pp_params, s._data.shape[0])}

    def _resolve_indices(self, params):
        indices = []
        for p in params:
            if p in self._PARAM_GROUPS:
                indices.extend(self._PARAM_GROUPS[p](self))
            elif isinstance(p, int):
                indices.append(p)
            else:
                raise ValueError(f"Unknown param '{p}'. Use 'focal', 'center', 'extra', or int.")
        return indices

    def fix(self, *params):
        with torch.no_grad():
            for i in self._resolve_indices(params):
                self._grad_mask[i] = 0.0
        return self

    def unfix(self, *params):
        with torch.no_grad():
            for i in self._resolve_indices(params):
                self._grad_mask[i] = 1.0
        return self

    @property
    def fixed(self):
        fe, pe = self.num_focal_params, self.num_focal_params + self.num_pp_params
        return {
            'focal': bool((self._grad_mask[:fe] == 0).all()),
            'center': bool((self._grad_mask[fe:pe] == 0).all()) if self.num_pp_params > 0 else True,
            'extra': bool((self._grad_mask[pe:] == 0).all()) if self.num_extra_params > 0 else True,
        }

    # ----- Abstract -----------------------------------------------------------

    def map(self, points3d):
        raise NotImplementedError

    def unmap(self, points2d):
        raise NotImplementedError

    def get_center(self):
        raise NotImplementedError

    def get_fov(self):
        raise NotImplementedError

    # ----- Tensor-like --------------------------------------------------------

    @property
    def image_shape(self):
        return self._image_shape

    def __repr__(self):
        fe, pe = self.num_focal_params, self.num_focal_params + self.num_pp_params
        fixed = ', '.join(k for k, v in self.fixed.items() if v)
        lines = [f'{self.model_name}: {{',
                 f'\timage_size: {self._image_shape.tolist()}',
                 f'\tfocals: {self._data[:fe].tolist()}',
                 f'\tpp: {self._data[fe:pe].tolist()}',
                 f'\textra: {self._data[pe:].tolist()}']
        if fixed:
            lines.append(f'\tfixed: {fixed}')
        return '\n'.join(lines) + '\n}'

    def __getitem__(self, idx):
        result = self._data[idx]
        mask = self._grad_mask[idx]
        if not mask.all():
            return result.detach() if (mask.ndim == 0 or not mask.any()) else result * mask + result.detach() * (1 - mask)
        return result

    def __setitem__(self, idx, value):
        with torch.no_grad():
            self._data[idx] = value

    def clone(self):
        new = type(self)(self._data.data.clone(), self._image_shape.clone())
        new._grad_mask.copy_(self._grad_mask)
        return new

    def detach(self):
        return type(self)(self._data.data.detach().clone(), self._image_shape.detach().clone())

    def check_bounds(self, pts2d):
        return (pts2d >= 0).all(dim=-1) & (pts2d[:, 0] <= self._image_shape[0] - 1) & (pts2d[:, 1] <= self._image_shape[1] - 1)

    def to_colmap(self):
        with torch.no_grad():
            fe, pe = self.num_focal_params, self.num_focal_params + self.num_pp_params
            parts = [self.model_name,
                     *map(str, self._image_shape.tolist()),
                     *map(str, self._data[:fe].tolist()),
                     *map(str, self._data[fe:pe].tolist()),
                     *map(str, self._data[pe:].tolist())]
        return ' '.join(parts)

    # ----- Properties ---------------------------------------------------------

    @property
    def dtype(self):  return self._data.dtype
    @property
    def device(self): return self._data.device
    @property
    def ndim(self):   return self._data.ndim
    @property
    def shape(self):  return self._data.shape
    @property
    def size(self):   return self._data.size()

    @property
    def requires_grad(self):
        return self._data.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._data.requires_grad_(value)
        if value and not hasattr(self, '_grad_hook_registered'):
            self._data.register_hook(self._mask_gradient)
            self._grad_hook_registered = True
