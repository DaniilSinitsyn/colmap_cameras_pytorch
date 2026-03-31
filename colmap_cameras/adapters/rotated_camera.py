"""
2026 Daniil Sinitsyn

Camera wrapper that applies an extrinsic rotation to rays.
"""
from .camera_adapter import CameraAdapter


class RotatedCamera(CameraAdapter):
    """Wraps a camera with a rotation matrix applied to rays.

    ``R_cam2world`` transforms rays from the inner camera's local frame
    into the world frame.  ``unmap`` returns world-frame rays;
    ``map`` expects world-frame points and rotates them into the camera
    frame before projecting.

    Usage::

        cam = RotatedCamera(inner_camera, R_cam2world)
        rays = cam.unmap(pts2d)        # rays in world frame
        pts2d, valid = cam.map(pts3d)  # pts3d in world frame
    """

    def __init__(self, inner, R_cam2world):
        super().__init__(inner)
        if R_cam2world.shape != (3, 3):
            raise ValueError(f"R_cam2world must be a 3x3 rotation matrix, got {R_cam2world.shape}")
        self.register_buffer('R_cam2world', R_cam2world)

    def unmap(self, points2d):
        result = self.inner.unmap(points2d)
        if isinstance(result, tuple):
            rays, valid = result
            return rays @ self.R_cam2world.T, valid
        return result @ self.R_cam2world.T

    def map(self, points3d):
        return self.inner.map(points3d @ self.R_cam2world)

    @property
    def model_name(self):
        return f"Rotated({self.inner.model_name})"
