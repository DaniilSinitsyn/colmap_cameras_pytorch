"""
2026 Daniil Sinitsyn

Camera wrapper that rescales 2D coordinates by a factor.
"""
from .camera_adapter import CameraAdapter


class ResizedCamera(CameraAdapter):
    """Wraps any camera and scales pixel coordinates.

    Usage::

        cam = ResizedCamera(inner, scale=0.5)   # half resolution
        cam = ResizedCamera(inner, scale=2.0)    # double resolution
    """

    def __init__(self, inner, scale):
        super().__init__(inner)
        self._scale = scale

    def map(self, pts3d):
        pts2d, valid = self.inner.map(pts3d)
        return pts2d * self._scale, valid

    def unmap(self, pts2d):
        return self.inner.unmap(pts2d / self._scale)

    @property
    def image_shape(self):
        return (self.inner.image_shape * self._scale).round()

    def get_center(self):
        return self.inner.get_center() * self._scale

    @property
    def model_name(self):
        return f"Resized({self.inner.model_name}, {self._scale})"
