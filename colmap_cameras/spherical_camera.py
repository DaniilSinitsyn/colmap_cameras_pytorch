"""
2026 Daniil Sinitsyn

Spherical camera base class for full-sphere projection models.
"""
from .base_model import BaseCamera


class SphericalCamera(BaseCamera):
    """Base class for spherical camera models (equirectangular, cubemap, etc.)."""

    def get_center(self):
        return self._image_shape.float() / 2.0
