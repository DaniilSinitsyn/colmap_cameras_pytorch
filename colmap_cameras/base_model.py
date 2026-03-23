"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import warnings


class BaseModel(torch.nn.Module):
    """
    Base camera model. Inherits from torch.nn.Module.

    Camera parameters are stored as a single nn.Parameter and can be
    selectively fixed via fix()/unfix(). Principal point is fixed by default.

    Example::

        cam = model_selector_from_str("OPENCV 640 480 500 500 320 240 0.1 -0.05 0 0")
        cam.fix('focal')
        cam.unfix('center')
        cam.fix(4, 5)               # fix individual params by index

        optimizer = torch.optim.Adam(cam.parameters(), lr=1e-3)
    """
    _image_shape: torch.Tensor
    _data: torch.nn.Parameter
    ROOT_FINDING_MAX_ITERATIONS = 20
    EPSILON = 1e-6

    def __init__(self, data, image_shape):
        super().__init__()

        if self.num_extra_params == -1:
            if data.shape[0] < self.num_focal_params + self.num_pp_params + 1:
                raise ValueError(
                    f"Expected at least {self.num_focal_params + self.num_pp_params + 1} "
                    f"parameters, got {data.shape[0]}"
                )
            self.num_extra_params = data.shape[0] - self.num_focal_params - self.num_pp_params

        expected = self.num_focal_params + self.num_pp_params + self.num_extra_params
        if data.shape[0] != expected:
            raise ValueError(f"Expected {expected} parameters, got {data.shape[0]}")

        if isinstance(data, torch.nn.Parameter):
            self._data = data
        else:
            self._data = torch.nn.Parameter(data, requires_grad=False)
        self.register_buffer('_image_shape', image_shape)
        self.register_buffer('_grad_mask', torch.ones(data.shape[0], dtype=data.dtype, device=data.device))

        if self.num_pp_params > 0:
            self.fix('center')

    # ----- Fixing / unfixing parameters ------------------------------------

    def _mask_gradient(self, grad):
        return grad * self._grad_mask

    def _resolve_indices(self, params):
        focal_end = self.num_focal_params
        pp_end = focal_end + self.num_pp_params
        total = self._data.shape[0]
        indices = []
        for p in params:
            if p == 'focal':
                indices.extend(range(0, focal_end))
            elif p == 'center':
                indices.extend(range(focal_end, pp_end))
            elif p == 'extra':
                indices.extend(range(pp_end, total))
            elif isinstance(p, int):
                if p < 0 or p >= total:
                    raise IndexError(f"Parameter index {p} out of range [0, {total})")
                indices.append(p)
            else:
                raise ValueError(
                    f"Unknown parameter specifier '{p}'. "
                    "Use 'focal', 'center', 'extra', or an integer index."
                )
        return indices

    def fix(self, *params):
        """Fix parameters so they receive zero gradient.

        Args:
            *params: 'focal', 'center', 'extra', or integer indices.
        """
        indices = self._resolve_indices(params)
        with torch.no_grad():
            for i in indices:
                self._grad_mask[i] = 0.0
        return self

    def unfix(self, *params):
        """Unfix parameters so they receive gradient again.

        Args:
            *params: 'focal', 'center', 'extra', or integer indices.
        """
        indices = self._resolve_indices(params)
        with torch.no_grad():
            for i in indices:
                self._grad_mask[i] = 1.0
        return self

    @property
    def fixed(self):
        """Dict showing which parameter groups are fully fixed."""
        focal_end = self.num_focal_params
        pp_end = focal_end + self.num_pp_params
        return {
            'focal': bool((self._grad_mask[:focal_end] == 0).all()),
            'center': bool((self._grad_mask[focal_end:pp_end] == 0).all()) if self.num_pp_params > 0 else True,
            'extra': bool((self._grad_mask[pp_end:] == 0).all()) if self.num_extra_params > 0 else True,
        }

    # ----- Projection interface --------------------------------------------

    def map(self, points3d):
        raise NotImplementedError

    def unmap(self, points2d):
        raise NotImplementedError

    # ----- Tensor-like interface -------------------------------------------

    def __repr__(self):
        f = f'{self.model_name}' + ': {'
        image_size = f'{self._image_shape.tolist()}'
        focals = f'{self._data[:self.num_focal_params].tolist()}'
        pp = f'{self._data[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()}'
        extra = f'{self._data[self.num_focal_params + self.num_pp_params:].tolist()}'
        fixed_info = ', '.join(k for k, v in self.fixed.items() if v)
        f += '\n\timage_size: ' + image_size
        f += '\n\tfocals: ' + focals
        f += '\n\tpp: ' + pp
        f += '\n\textra: ' + extra
        if fixed_info:
            f += '\n\tfixed: ' + fixed_info
        f += '\n}'
        return f

    @property
    def image_shape(self):
        return self._image_shape

    def check_bounds(self, points2d):
        return (
            (points2d >= 0).all(dim=-1)
            & (points2d[:, 0] <= self._image_shape[0] - 1)
            & (points2d[:, 1] <= self._image_shape[1] - 1)
        )

    def clone(self):
        new = type(self)(self._data.data.clone(), self._image_shape.clone())
        new._grad_mask.copy_(self._grad_mask)
        return new

    def __getitem__(self, idx):
        result = self._data[idx]
        mask = self._grad_mask[idx]
        if not mask.all():
            if mask.ndim == 0 or not mask.any():
                return result.detach()
            result = result * mask + result.detach() * (1 - mask)
        return result

    def __setitem__(self, idx, value):
        with torch.no_grad():
            self._data[idx] = value

    def detach(self):
        return type(self)(self._data.data.detach().clone(), self._image_shape.detach().clone())

    def to_colmap(self):
        with torch.no_grad():
            sh = list(map(str, self._image_shape.tolist()))
            fp = list(map(str, self._data[:self.num_focal_params].tolist()))
            pp = list(map(str, self._data[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()))
            ep = list(map(str, self._data[self.num_focal_params + self.num_pp_params:].tolist()))
        f = f'{self.model_name} ' + ' '.join(sh) + ' '
        f += ' '.join(fp) + ' ' + ' '.join(pp) + ' ' + ' '.join(ep)
        return f

    def get_focal(self):
        return self._data[:self.num_focal_params].mean()

    def get_center(self):
        return self._data[self.num_focal_params:self.num_focal_params + self.num_pp_params]

    def set_focal(self, value):
        with torch.no_grad():
            self._data[:self.num_focal_params] = value

    def set_center(self, value):
        with torch.no_grad():
            self._data[self.num_focal_params:self.num_focal_params + self.num_pp_params] = value

    def get_center_resolution_focal(self):
        focal = 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                point = self.get_center()
                dxy = torch.tensor([dx, dy]).to(point)
                dxy = dxy / torch.linalg.norm(dxy)
                ray = self.unmap(point + dxy).reshape(-1)
                ray = ray / torch.linalg.norm(ray)
                phi = torch.atan2(torch.linalg.norm(ray[:2]), ray[2])
                focal += 0.125 / phi
        return focal

    def initialize_from(self, other_model, nonlinear_iterations=20):
        self.set_focal(other_model.get_center_resolution_focal())
        self.set_center(other_model.get_center())

        n_points = int((self.image_shape[0] - self.get_center()[0]).item())
        steps = torch.linspace(0, n_points, n_points).to(self.device)
        pts2d = self.get_center() + steps.reshape(-1, 1) * torch.tensor([0.7, 0.7], device=self.device)
        pts2d = pts2d.to(other_model.device)
        pts3d = other_model.unmap(pts2d)
        valid = ~torch.isnan(pts3d).any(dim=-1)
        pts2d = pts2d[valid]
        pts3d = pts3d[valid]

        self.initialize_distortion_from_points(pts2d, pts3d)
        self.__initialize_from_nonlinear(pts2d.detach(), pts3d.detach(), nonlinear_iterations)

    def initialize_distortion_from_points(self, pts2d, pts3d):
        warning = f"initialize_distortion_from_points is not implemented for {self.model_name} model"
        warnings.warn(warning)

    def __initialize_from_nonlinear(self, pts2d, pts3d, iterations):
        self.requires_grad = True
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        for i in range(iterations):
            optimizer.zero_grad()
            proj, valid = self.map(pts3d)
            valid &= ~torch.isnan(proj).all(dim=-1)
            if valid.sum() == 0:
                continue
            err = ((proj[valid] - pts2d[valid]) ** 2).mean()
            err.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if i % max(1, iterations // 5) == 0:
                print(f'Iteration {i}/{iterations} : Mean reprojection error {err.item()}')

        self.requires_grad = False

    # ----- Properties ------------------------------------------------------

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return self._data.device

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size()

    @property
    def requires_grad(self):
        return self._data.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self._data.requires_grad_(value)
        if value and not hasattr(self, '_grad_hook_registered'):
            self._data.register_hook(self._mask_gradient)
            self._grad_hook_registered = True
