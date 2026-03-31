"""
2026 Daniil Sinitsyn

A collection of named cameras that projects world-frame points to each camera.
"""
import torch


class CameraRig(torch.nn.Module):
    """A rig of named cameras that projects world-frame 3D points into
    each camera's pixel space.

    Cameras are stored as a dict and accessed by name.

    Usage::

        rig = CameraRig({
            'front': MaskedCamera(ValidatedCamera(front_model), border_size=50),
            'back':  MaskedCamera(ValidatedCamera(RotatedCamera(back_model, R)), border_size=50),
        })
        results = rig.map(pts3d)
        # results['front'] = (pts2d, valid)
        # results['back']  = (pts2d, valid)
    """

    def __init__(self, cameras):
        super().__init__()
        self.cameras = torch.nn.ModuleDict(cameras)

    def map(self, points3d):
        """Project world-frame points through each camera.

        Returns a dict mapping camera name to (pts2d, valid) tuples.
        """
        return {name: cam.map(points3d) for name, cam in self.cameras.items()}

    def __getitem__(self, name):
        return self.cameras[name]
