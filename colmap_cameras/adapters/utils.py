"""
2026 Daniil Sinitsyn

Utility functions for camera adapters.
"""
from .camera_adapter import CameraAdapter


def has_adapter(cam, adapter_cls):
    """Check if a camera's adapter chain contains an instance of adapter_cls."""
    while isinstance(cam, CameraAdapter):
        if isinstance(cam, adapter_cls):
            return True
        cam = cam.inner
    return False
