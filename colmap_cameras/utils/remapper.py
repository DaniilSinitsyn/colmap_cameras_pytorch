"""
2026 Daniil Sinitsyn

Camera model image remapping.
"""
import torch
import cv2
import numpy as np
from ..models import SimplePinhole


class Remapper:

    def __init__(self, step=1):
        self.step = step

    def compute_lut(self, model_in, model_out):
        """Compute remap LUTs from model_out pixel space to model_in pixel space."""
        w, h = [int(x.item()) for x in model_out.image_shape]
        device = model_out.device

        ud, vd = torch.meshgrid(
            torch.arange(0, w, self.step, device=device),
            torch.arange(0, h, self.step, device=device), indexing='xy')

        pts = torch.stack([ud.ravel(), vd.ravel()], dim=-1)
        pts3d = model_out.unmap(pts.float())
        pts2d, valid = model_in.map(pts3d)
        pts2d[~valid] = -1

        ws, hs = w // self.step, h // self.step
        pts2d = pts2d.reshape(hs, ws, 2)
        return pts2d[..., 0].cpu().numpy().astype(np.float32), \
               pts2d[..., 1].cpu().numpy().astype(np.float32)

    def remap(self, model_in, model_out, img):
        w, h = [int(x.item()) for x in model_out.image_shape]
        xlut, ylut = self.compute_lut(model_in, model_out)
        if isinstance(img, str):
            img = cv2.imread(img)
        return cv2.resize(cv2.remap(img, xlut, ylut, cv2.INTER_LINEAR), (w, h))

    def remap_from_fov(self, model_in, fov_out, img):
        model_out = SimplePinhole.from_fov(fov_out, model_in.image_shape).to(model_in.device)
        return self.remap(model_in, model_out, img)
