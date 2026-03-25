"""
2026 Daniil Sinitsyn

Perspective camera base class with focal length, principal point, and distortion.
"""
import torch
import warnings

from .base_model import BaseCamera


class PerspectiveCamera(BaseCamera):

    def __init__(self, data, image_shape):
        super().__init__(data, image_shape)
        if self.num_pp_params > 0:
            self.fix('center')

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

    def get_fov(self):
        f = self.get_focal()
        return torch.stack([2 * torch.atan(self._image_shape[0] / (2 * f)),
                            2 * torch.atan(self._image_shape[1] / (2 * f))])

    def get_center_resolution_focal(self):
        focal = 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                pt = self.get_center()
                dxy = torch.tensor([dx, dy], dtype=pt.dtype, device=pt.device)
                dxy = dxy / dxy.norm()
                ray = self.unmap(pt + dxy).reshape(-1)
                ray = ray / ray.norm()
                focal += 0.125 / torch.atan2(ray[:2].norm(), ray[2])
        return focal

    def initialize_from(self, other_model, nonlinear_iterations=20):
        self.set_focal(other_model.get_center_resolution_focal())
        self.set_center(other_model.get_center())

        n = int((self.image_shape[0] - self.get_center()[0]).item())
        steps = torch.linspace(0, n, n, device=self.device)
        pts2d = self.get_center() + steps[:, None] * torch.tensor([0.7, 0.7], device=self.device)
        pts3d = other_model.unmap(pts2d.to(other_model.device))
        valid = ~torch.isnan(pts3d).any(dim=-1)

        self.initialize_distortion_from_points(pts2d[valid], pts3d[valid])
        self._initialize_from_nonlinear(pts2d[valid].detach(), pts3d[valid].detach(), nonlinear_iterations)

    def initialize_distortion_from_points(self, pts2d, pts3d):
        warnings.warn(f"initialize_distortion_from_points not implemented for {self.model_name}")

    def _initialize_from_nonlinear(self, pts2d, pts3d, iterations):
        self.requires_grad = True
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
        for i in range(iterations):
            opt.zero_grad()
            proj, valid = self.map(pts3d)
            valid &= ~torch.isnan(proj).all(dim=-1)
            if valid.sum() == 0:
                continue
            err = ((proj[valid] - pts2d[valid]) ** 2).mean()
            err.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            if i % max(1, iterations // 5) == 0:
                print(f'Iteration {i}/{iterations} : Mean reprojection error {err.item()}')
        self.requires_grad = False
