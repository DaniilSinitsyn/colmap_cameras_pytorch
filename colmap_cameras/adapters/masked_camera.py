"""
2026 Daniil Sinitsyn

Camera wrapper that builds a pixel mask combining validity and border exclusion.
"""
import numpy as np

from .camera_adapter import CameraAdapter
from ..utils.valid_region import estimate_valid_region


class MaskedCamera(CameraAdapter):
    """Wraps a camera and builds a uint8 image mask combining the valid
    region (from round-trip Jacobian check) with a pixel border exclusion.

    The mask is a numpy array (h, w) with 255 for valid pixels and 0
    elsewhere, suitable for cv2.remap and saving to disk.

    A pre-computed mask can be passed directly to skip estimation.

    Usage::

        cam = MaskedCamera(inner, border_size=50)           # estimate valid region
        cam = MaskedCamera(inner, border_size=50, mask=m)   # use provided mask
        cam.mask          # (h, w) uint8 numpy array
    """

    def __init__(self, inner, border_size, mask=None, step=2.0):
        super().__init__(inner)
        if border_size < 0:
            raise ValueError(f"border_size must be non-negative, got {border_size}")
        self.border_size = border_size

        if mask is not None:
            valid = mask.copy()
        else:
            valid = estimate_valid_region(inner, step=step).cpu().numpy().astype(np.uint8) * 255

        if border_size > 0:
            valid[:border_size, :] = 0
            valid[-border_size:, :] = 0
            valid[:, :border_size] = 0
            valid[:, -border_size:] = 0

        self.mask = valid

    def check_bounds(self, pts2d):
        """Check that points lie inside the image minus the border."""
        w, h = self.image_shape[0], self.image_shape[1]
        b = self.border_size
        return (
            (pts2d[:, 0] >= b) &
            (pts2d[:, 1] >= b) &
            (pts2d[:, 0] <= w - 1 - b) &
            (pts2d[:, 1] <= h - 1 - b)
        )

    def map(self, points3d):
        pts2d, valid = self.inner.map(points3d)
        valid = valid & self.check_bounds(pts2d)
        pts2d[~valid] = 0
        return pts2d, valid

    @property
    def model_name(self):
        return f"Masked({self.inner.model_name})"
