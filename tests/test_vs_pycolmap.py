"""
Compare our camera models against pycolmap's reference implementation.
"""
import unittest
import torch
import numpy as np

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

from colmap_cameras import model_selector_from_str

# (colmap_string, pycolmap_model_id)
TEST_CAMERAS = [
    ("SIMPLE_PINHOLE 640 480 500 320 240",
     pycolmap.CameraModelId.SIMPLE_PINHOLE if HAS_PYCOLMAP else None),
    ("PINHOLE 640 480 500 500 320 240",
     pycolmap.CameraModelId.PINHOLE if HAS_PYCOLMAP else None),
    ("SIMPLE_RADIAL 640 480 500 320 240 0.1",
     pycolmap.CameraModelId.SIMPLE_RADIAL if HAS_PYCOLMAP else None),
    ("RADIAL 640 480 500 320 240 0.1 -0.05",
     pycolmap.CameraModelId.RADIAL if HAS_PYCOLMAP else None),
    ("OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002",
     pycolmap.CameraModelId.OPENCV if HAS_PYCOLMAP else None),
    ("FULL_OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002 0.01 -0.02 0.005 -0.001",
     pycolmap.CameraModelId.FULL_OPENCV if HAS_PYCOLMAP else None),
    ("FOV 640 480 500 500 320 240 0.8",
     pycolmap.CameraModelId.FOV if HAS_PYCOLMAP else None),
    ("SIMPLE_RADIAL_FISHEYE 640 480 500 320 240 0.1",
     pycolmap.CameraModelId.SIMPLE_RADIAL_FISHEYE if HAS_PYCOLMAP else None),
    ("RADIAL_FISHEYE 640 480 500 320 240 0.1 -0.05",
     pycolmap.CameraModelId.RADIAL_FISHEYE if HAS_PYCOLMAP else None),
    ("OPENCV_FISHEYE 640 480 500 500 320 240 0.1 -0.05 0.01 -0.005",
     pycolmap.CameraModelId.OPENCV_FISHEYE if HAS_PYCOLMAP else None),
    ("THIN_PRISM_FISHEYE 640 480 500 500 320 240 0.1 -0.05 0.001 0.002 0.01 -0.02 0.005 -0.001",
     pycolmap.CameraModelId.THIN_PRISM_FISHEYE if HAS_PYCOLMAP else None),
]


def make_pycolmap_camera(colmap_str, model_id):
    parts = colmap_str.split()
    w, h = int(float(parts[1])), int(float(parts[2]))
    params = [float(x) for x in parts[3:]]
    cam = pycolmap.Camera()
    cam.model = model_id
    cam.width = w
    cam.height = h
    cam.params = params
    return cam


@unittest.skipUnless(HAS_PYCOLMAP, "pycolmap not installed")
class TestVsPycolmap(unittest.TestCase):

    def _test_map(self, colmap_str, model_id):
        ours = model_selector_from_str(colmap_str)
        ref = make_pycolmap_camera(colmap_str, model_id)

        # Generate test points in normalized camera coordinates
        u = torch.arange(-0.3, 0.3, 0.05)
        v = torch.arange(-0.3, 0.3, 0.05)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        pts3d = torch.stack([uu.ravel(), vv.ravel(), torch.ones_like(uu.ravel())], dim=-1).float()

        # Our projection
        pts2d_ours, valid = ours.map(pts3d)

        # pycolmap projection (works on 2D normalized coords)
        pts2d_ref = []
        for p in pts3d[valid].numpy():
            xy = p[:2] / p[2]
            pts2d_ref.append(ref.img_from_cam(xy))
        pts2d_ref = torch.tensor(np.array(pts2d_ref))

        diff = (pts2d_ours[valid] - pts2d_ref).abs()
        max_err = diff.max().item()
        self.assertLess(max_err, 1e-4, f"{colmap_str.split()[0]} map max_err={max_err}")

    def _test_unmap(self, colmap_str, model_id):
        ours = model_selector_from_str(colmap_str)
        ref = make_pycolmap_camera(colmap_str, model_id)

        # Generate test pixel coordinates near center
        cx, cy = float(ref.principal_point_x), float(ref.principal_point_y)
        u = torch.arange(cx - 100, cx + 100, 20)
        v = torch.arange(cy - 100, cy + 100, 20)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1).float()

        # Our unprojection (returns 3D with z=1 convention)
        pts3d_ours = ours.unmap(pts2d)
        xy_ours = pts3d_ours[:, :2] / pts3d_ours[:, 2:3]

        # pycolmap unprojection (returns 2D normalized coords)
        xy_ref = []
        for p in pts2d.numpy():
            xy_ref.append(ref.cam_from_img(p))
        xy_ref = torch.tensor(np.array(xy_ref))

        diff = (xy_ours - xy_ref).abs()
        max_err = diff.max().item()
        self.assertLess(max_err, 1e-4, f"{colmap_str.split()[0]} unmap max_err={max_err}")

    def test_map_all(self):
        for colmap_str, model_id in TEST_CAMERAS:
            with self.subTest(model=colmap_str.split()[0]):
                self._test_map(colmap_str, model_id)

    def test_unmap_all(self):
        for colmap_str, model_id in TEST_CAMERAS:
            with self.subTest(model=colmap_str.split()[0]):
                self._test_unmap(colmap_str, model_id)


if __name__ == '__main__':
    unittest.main()
