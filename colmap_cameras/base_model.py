"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import warnings

class BaseModel:
    """
    Partially copilot generated class that behaves almost like a torch.Tensor
    """
    _image_shape : torch.Tensor
    _data : torch.Tensor
    ROOT_FINDING_MAX_ITERATIONS = 100
    OPTIMIZATION_FIX_FOCALS = False
    OPTIMIZATION_FIX_CENTER = True
    OPTIMIZATION_FIX_EXTRA = False
    EPSILON = 1e-6

    def __init__(self, data, image_shape):
        self._data = data
        self._image_shape = image_shape
        if self.num_extra_params == -1:
            if data.shape[0] < self.num_focal_params + self.num_pp_params + 1:
                raise ValueError(f"Expected at least {self.num_focal_params + self.num_pp_params + 1} parameters, got {data.shape[0]}")
            self.num_extra_params = data.shape[0] - self.num_focal_params - self.num_pp_params

        if data.shape[0] != self.num_focal_params + self.num_pp_params + self.num_extra_params:
            raise ValueError(f"Expected {self.num_focal_params + self.num_pp_params + self.num_extra_params} parameters, got {data.shape[0]}")
    
    def __repr__(self):
        f = f'{self.model_name}' + ': {'
        image_size = f'{self._image_shape.tolist()}'
        focals = f'{self[:self.num_focal_params].tolist()}'
        pp = f'{self[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()}'
        extra = f'{self[self.num_focal_params + self.num_pp_params:].tolist()}'
        f = f + '\n\timage_size: ' + image_size
        f = f + '\n\tfocals: ' + focals
        f = f + '\n\tpp: ' + pp
        f = f + '\n\textra: ' + extra + '\n}'
        return f

    def map(self, points3d):
        raise NotImplementedError

    def unmap(self, points2d):
        raise NotImplementedError

    @property
    def image_shape(self):
        return self._image_shape
    
    def check_bounds(self, points2d):
        return (points2d >= 0).all(dim=-1) & (points2d[:, 0] <= self._image_shape[0]-1) & (points2d[:, 1] <= self._image_shape[1]-1)   
    def clone(self, *args, **kwargs):
        return type(self)(self._data.clone(*args, **kwargs), self._image_shape.clone(*args, **kwargs))
    
    def __getitem__(self, idx):
        focal_end = self.num_focal_params
        pp_end = focal_end + self.num_pp_params
        if self.OPTIMIZATION_FIX_CENTER or self.OPTIMIZATION_FIX_FOCALS or self.OPTIMIZATION_FIX_EXTRA:
            if type(idx) == slice:
                if idx.start == None: idx = slice(0, idx.stop)
                if idx.stop == None: idx = slice(idx.start, self._data.shape[0])
                
                if self.OPTIMIZATION_FIX_FOCALS and idx.start < focal_end and idx.stop <= focal_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_CENTER and idx.start >= focal_end and idx.stop <= pp_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_EXTRA and idx.start >= pp_end and idx.stop <= self._data.shape[0]:
                    return self._data[idx].detach()

            else:
                if self.OPTIMIZATION_FIX_FOCALS and idx < focal_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_CENTER and idx >= focal_end and idx < pp_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_EXTRA and idx >= pp_end and idx < self._data.shape[0]:
                    return self._data[idx].detach()

        return  self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    def cpu(self):
        return type(self)(self._data.cpu(), self._image_shape.cpu())
    
    def cuda(self, device=None):
        return type(self)(self._data.cuda(device), self._image_shape.cuda(device))

    def detach(self):
        return type(self)(self._data.detach(), self._image_shape.detach())

    def to(self, *args, **kwargs):
        return type(self)(self._data.to(*args, **kwargs), self._image_shape.to(*args, **kwargs))
    
    def to_colmap(self):
        sh = list(map(str, self._image_shape.tolist()))
        fp = list(map(str, self[:self.num_focal_params].tolist()))
        pp = list(map(str, self[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()))
        ep = list(map(str, self[self.num_focal_params + self.num_pp_params:].tolist()))
        f = f'{self.model_name} ' + ' '.join(sh) + ' '
        f = f + ' '.join(fp) + ' ' + ' '.join(pp) + ' ' + ' '.join(ep)
        return f
    
    def get_focal(self):
        return self[:self.num_focal_params].mean()
    def get_center(self):
        return self[self.num_focal_params:self.num_focal_params + self.num_pp_params]
    
    def set_focal(self, value):
        self[:self.num_focal_params] = value

    def set_center(self, value):
        self[self.num_focal_params:self.num_focal_params + self.num_pp_params] = value

    def get_center_resolution_focal(self):
        focal = 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
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
        self.__initialize_from_nonlinear(pts2d, pts3d, nonlinear_iterations)
    
    def initialize_distortion_from_points(self, pts2d, pts3d):
        warning = f"initialize_distortion_from_points is not implemented for {self.model_name} model"
        warnings.warn(warning)

    def __initialize_from_nonlinear(self, pts2d, pts3d, iterations):
        for i in range(iterations):
            proj, valid = self.map(pts3d)
            valid &= ~torch.isnan(proj).all(dim=-1)
        
            if valid.sum() == 0: continue
            def res(camera_data):
                new_model = self.clone()
                new_model._data = camera_data
                pts2d_, _ = new_model.map(pts3d)
                return pts2d_
                
            J = torch.autograd.functional.jacobian(res, self._data, vectorize=True)
            err = proj[valid] - pts2d[valid]
            J = J[valid]
            J = J.reshape(err.shape[0], 2, -1)

            H = (J.transpose(1, 2) @ J).mean(dim=0)
            b = (J.transpose(1, 2) @ err[...,None]).mean(dim=0)
            
            H += 1e-6 * torch.eye(H.shape[0]).to(H)
            step = torch.linalg.lstsq(H, b)[0].squeeze()
            if torch.isnan(step).any() or torch.isinf(step).any():
                break
            self._data = self._data - step
            err = torch.linalg.norm(err, dim=-1).mean()

            print(f'Iteration {i}/{iterations} : Mean reprojection error {err.item()}')

    @property
    def dtype(self): return self._data.dtype
    @property
    def device(self): return self._data.device
    @property
    def ndim(self): return self._data.ndim
    @property
    def shape(self): return self._data.shape
    @property
    def size(self): return self._data.size()

    def get_requires_grad(self):
        return self._data.requires_grad
    def set_requires_grad(self, value: bool):
        self._data.requires_grad = value
    requires_grad = property(get_requires_grad, set_requires_grad)
