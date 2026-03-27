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

    def rescale(self, scale):
        """Create a new camera with rescaled focal, center, and image size."""
        data = self._data.data.clone()
        fe = self.num_focal_params
        pe = fe + self.num_pp_params
        data[:fe] *= scale
        data[fe:pe] *= scale
        image_shape = (self._image_shape * scale).round()
        new = type(self)(data, image_shape)
        new._grad_mask.copy_(self._grad_mask)
        return new

    def get_fov(self):
        f = self.get_focal()
        return torch.stack([2 * torch.atan(self._image_shape[0] / (2 * f)),
                            2 * torch.atan(self._image_shape[1] / (2 * f))])

    def get_center_resolution_focal(self):
        """Effective focal length in pixels at the center.
        For standard perspective models this is just the stored focal.
        Models where the stored focal has different semantics (e.g. UCM, Mei)
        override this with a numerical estimate."""
        return self.get_focal()

    def _estimate_center_resolution_focal(self):
        """Numerically estimate focal from d(pixel)/d(angle) at the center."""
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

    def initialize_from(self, other_model, nonlinear_iterations=20, lr=0.01):
        self.set_focal(other_model.get_center_resolution_focal())
        self.set_center(other_model.get_center())

        w, h = [int(x.item()) for x in other_model.image_shape]
        sample_step = max(1, min(w, h) // 100)
        uu, vv = torch.meshgrid(
            torch.arange(0, w, sample_step, device=self.device, dtype=torch.float32),
            torch.arange(0, h, sample_step, device=self.device, dtype=torch.float32),
            indexing='xy')
        pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)

        with torch.no_grad():
            pts3d = other_model.unmap(pts2d.to(other_model.device))
            ok = ~torch.isnan(pts3d).any(dim=-1)
            # Filter by round-trip error — catches folds, bad roots, everything
            if ok.any():
                rt, rt_v = other_model.map(pts3d[ok])
                rt_err = (rt - pts2d[ok]).pow(2).sum(dim=-1).sqrt()
                ok[ok.clone()] = rt_v & (rt_err < 1.0)

        # Normalize rays — projection only depends on direction, keeps Jacobian bounded
        pts3d_ok = pts3d[ok]
        pts3d_ok = pts3d_ok / pts3d_ok.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        self.initialize_distortion_from_points(pts2d[ok], pts3d_ok)
        self._initialize_from_nonlinear(pts2d[ok].detach(), pts3d_ok.detach(), nonlinear_iterations, lr)

    def initialize_distortion_from_points(self, pts2d, pts3d):
        warnings.warn(f"initialize_distortion_from_points not implemented for {self.model_name}")

    def _initialize_from_nonlinear(self, pts2d, pts3d, iterations, lr=0.01):
        P = self._data.shape[0]
        damping = 1e-2
        prev_err = float('inf')

        for i in range(iterations):
            if self.CLOSED_FORM_MAP:
                pred, valid, J = self.map_with_jac(pts3d)
                valid &= ~torch.isnan(pred).any(dim=-1)
                if valid.sum() == 0:
                    continue
                r = (pred[valid] - pts2d[valid]).reshape(-1)
                Jv = J[valid].reshape(-1, P)
            else:
                pred, J = self.unmap_with_jac(pts2d)
                valid = ~torch.isnan(pred).any(dim=-1)
                pts3d_n = pts3d / pts3d.norm(dim=-1, keepdim=True)
                pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if valid.sum() == 0:
                    continue
                r = (pred_n[valid] - pts3d_n[valid]).reshape(-1)
                Jv = J[valid].reshape(-1, P)

            # Zero out Jacobian columns for fixed params
            fixed_mask = (self._grad_mask == 0)
            if fixed_mask.any():
                Jv[:, fixed_mask] = 0

            err = (r ** 2).mean().item()

            # Levenberg-Marquardt: adapt damping
            if err < prev_err:
                damping = max(damping * 0.5, 1e-6)
            else:
                damping = min(damping * 5.0, 1e4)
            prev_err = err

            JtJ = Jv.T @ Jv
            JtJ += damping * JtJ.diag().clamp(min=1e-6).diag()
            Jtr = Jv.T @ r
            try:
                delta = torch.linalg.solve(JtJ, Jtr)
            except torch._C._LinAlgError:
                delta = torch.linalg.lstsq(JtJ, Jtr.unsqueeze(-1)).solution.squeeze(-1)

            with torch.no_grad():
                self._data -= delta * self._grad_mask

            if i % max(1, iterations // 5) == 0:
                print(f'Iteration {i}/{iterations} : Mean reprojection error {err:.6f}')
