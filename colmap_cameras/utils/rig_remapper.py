"""
2026 Daniil Sinitsyn

Remaps images from a CameraRig into a target camera's pixel space.
"""
import numpy as np
import torch
import cv2

from ..adapters.utils import has_adapter
from ..adapters.masked_camera import MaskedCamera
from ..adapters.lut_camera import LUTCamera


class RigRemapper:
    """Remaps images from a CameraRig into a target camera's pixel space.

    Ensures each source camera is wrapped in MaskedCamera and LUTCamera.
    Uses distance-based blending: each camera's weight is proportional to
    its distance from the valid region boundary.

    Usage::

        rig = CameraRig({'front': front_cam, 'back': back_cam})
        remapper = RigRemapper(rig)

        # One-off
        image, mask = remapper.remap({'front': img_f, 'back': img_b}, target)

        # Batch — precompute once, apply many times
        luts = remapper.build_luts(target)
        for img_f, img_b in pairs:
            image, mask = RigRemapper.remap_from_luts({'front': img_f, 'back': img_b}, luts)
    """

    def __init__(self, rig, mask_border=0):
        """
        Args:
            rig: CameraRig with named cameras.
            mask_border: pixels to erode each mask in target space.
                Creates a zero-gap of 2*mask_border between adjacent cameras.
        """
        for name, cam in rig.cameras.items():
            if not has_adapter(cam, MaskedCamera):
                cam = MaskedCamera(cam, border_size=0)
            if not has_adapter(cam, LUTCamera):
                cam = LUTCamera(cam)
            rig.cameras[name] = cam
        self.rig = rig
        self.mask_border = mask_border

    def build_luts(self, target):
        """Build remap LUTs, exclusive masks, and distance-based blend weights.

        Returns:
            dict of {name: (map_x, map_y, mask, weight)} where
            map_x, map_y: numpy float32 remap arrays
            mask:   numpy uint8 (0/255), eroded by mask_border
            weight: numpy float32 (0..1), distance-based blend weight
        """
        w, h = int(target.image_shape[0].item()), int(target.image_shape[1].item())

        u = torch.arange(w, device=target.device, dtype=torch.float32)
        v = torch.arange(h, device=target.device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        rays = target.unmap(torch.stack([uu.ravel(), vv.ravel()], dim=-1))
        if isinstance(rays, tuple):
            rays = rays[0]

        raw_masks = {}
        map_xy = {}
        with torch.no_grad():
            for name, cam in self.rig.cameras.items():
                pts2d, valid = cam.map(rays)

                map_x = pts2d[:, 0].cpu().numpy().reshape(w, h).T.astype(np.float32)
                map_y = pts2d[:, 1].cpu().numpy().reshape(w, h).T.astype(np.float32)
                invalid = ~valid.cpu().numpy().reshape(w, h).T
                map_x[invalid] = -1
                map_y[invalid] = -1

                map_xy[name] = (map_x, map_y)
                raw_masks[name] = cv2.remap(
                    cam.mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0,
                )

        # Distance-based weights
        distances = {
            name: cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            for name, mask in raw_masks.items()
        }
        total = np.sum(list(distances.values()), axis=0)
        total[total == 0] = 1.0
        weights = {name: dist / total for name, dist in distances.items()}

        # Erode masks
        luts = {}
        for name in self.rig.cameras:
            mask = raw_masks[name].copy()
            if self.mask_border > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * self.mask_border + 1, 2 * self.mask_border + 1),
                )
                mask = cv2.erode(mask, kernel)
            map_x, map_y = map_xy[name]
            luts[name] = (map_x, map_y, mask, weights[name])

        return luts

    def remap(self, images, target):
        """Remap and blend source images into the target camera.

        Builds LUTs on the fly. For batch processing, use build_luts()
        once and then remap_from_luts() for each frame.

        Returns:
            (image, mask) — blended uint8 image and combined uint8 mask.
        """
        return self.remap_from_luts(images, self.build_luts(target))

    @staticmethod
    def remap_from_luts(images, luts):
        """Remap using precomputed LUTs from build_luts().

        Args:
            images: dict of {name: numpy BGR image}.
            luts: output of build_luts().

        Returns:
            (image, mask) — blended uint8 image and combined uint8 mask.
        """
        first = next(iter(luts.values()))
        h, w = first[2].shape[:2]

        result = np.zeros((h, w, 3), dtype=np.float32)
        result_mask = np.zeros((h, w), dtype=np.uint8)

        for name, (map_x, map_y, mask, weight) in luts.items():
            remapped = cv2.remap(images[name], map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
            result += remapped.astype(np.float32) * weight[:, :, None]
            result_mask = np.maximum(result_mask, mask)

        return result.astype(np.uint8), result_mask
