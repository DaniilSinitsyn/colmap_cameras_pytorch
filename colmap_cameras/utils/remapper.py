"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import cv2
import numpy as np
from ..models import SimplePinhole


class Remapper:

    def __init__(self, step=1):
        self.step = step

    def remap(self, model_in, model_out, img):
        w_i, h_i = [int(x.item()) for x in model_out.image_shape]
        device = model_out.device
        assert model_in.device == device, "Models should be on the same device"

        ud, vd = torch.meshgrid(torch.arange(0, w_i, self.step, device=device),
                                torch.arange(0, h_i, self.step, device=device),
                                indexing='xy')
        w = w_i // self.step
        h = h_i // self.step

        points = torch.stack([ud.ravel(), vd.ravel()], dim=-1)
        points3d = model_out.unmap(points.float())
        points2d, valid = model_in.map(points3d)

        points2d[~valid] = -1
        points2d = points2d.reshape(h, w, 2)

        xlut = points2d[..., 0].cpu().numpy()
        ylut = points2d[..., 1].cpu().numpy()
        
        if isinstance(img, str):
            img = cv2.imread(img)

        img = cv2.remap(img, xlut, ylut, cv2.INTER_LINEAR)
        img = cv2.resize(img, (w_i, h_i))
        return img
    
    def remap_from_fov(self, model_in, fov_out, img):
        model_out = SimplePinhole.from_fov(fov_out, model_in.image_shape).to(model_in.device)
        return self.remap(model_in, model_out, img)
