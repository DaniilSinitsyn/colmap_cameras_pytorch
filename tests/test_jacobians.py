"""
2026 Daniil Sinitsyn

Tests for map_with_jac / unmap_with_jac and Gauss-Newton fitting.
"""
import unittest
import torch
from colmap_cameras import model_selector_from_str


# Test cameras with various model types
TEST_CAMERAS = [
    'SIMPLE_PINHOLE 200 200 100 100 100',
    'PINHOLE 200 200 100 100 100 100',
    'SIMPLE_RADIAL 200 200 100 100 100 -0.3',
    'RADIAL 200 200 100 100 100 0.1 -0.02',
    'OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002',
    'FULL_OPENCV 640 480 300 300 320 240 0.1 -0.05 0.01 -0.005 0.001 0.0 0.0 0.0',
    'OPENCV_FISHEYE 640 480 300 300 320 240 0.1 -0.05 0.01 -0.005',
    'SIMPLE_RADIAL_FISHEYE 200 200 100 100 100 0.01',
    'RADIAL_FISHEYE 200 200 100 100 100 0.01 -0.005',
    'FOV 640 480 500 500 320 240 0.8',
    'THIN_PRISM_FISHEYE 640 480 300 300 320 240 0.1 -0.05 0.001 0.002 0.01 -0.005 0.0001 0.0002',
    'UNIFIED_CAMERA_MODEL 200 200 100 100 100 100 0.5',
    'DIVISION_MODEL 200 200 100 100 100 0.1',
    'MEIS_CAMERA_MODEL 200 200 140 140 100 100 0.5 0.01 0.001 0.0001 0.0001',
    'MEIS_EXTENDED_CAMERA_MODEL 200 200 140 140 100 100 0.5 0.01 0.001 0.0001 0.0001 0.0001',
    'WOODSCAPE 640 480 300 320 240 1.0 -0.1 0.01 -0.001',
]

TEST_CAMERAS_UNMAP = [
    'POLYNOMIAL_DIVISION_MODEL 200 200 100 100 100 0.01 -0.001',
]


def make_pts3d(n=50):
    """Points in front of the camera, moderate angles."""
    torch.manual_seed(42)
    pts = torch.randn(n, 3)
    pts[:, 2] = pts[:, 2].abs() + 1.0
    return pts


def make_pts2d(cam, n=50):
    """Valid pixels spread across the image, avoiding exact center."""
    w, h = [int(x.item()) for x in cam.image_shape]
    torch.manual_seed(42)
    u = w * 0.2 + torch.rand(n) * w * 0.6
    v = h * 0.2 + torch.rand(n) * h * 0.6
    return torch.stack([u, v], dim=-1)


class TestMapWithJac(unittest.TestCase):

    def test_jacobian_shape(self):
        for cam_str in TEST_CAMERAS:
            cam = model_selector_from_str(cam_str)
            pts3d = make_pts3d(10)
            pts2d, valid, J = cam.map_with_jac(pts3d)
            P = cam._data.shape[0]
            self.assertEqual(J.shape, (10, 2, P), msg=cam_str)

    def test_jacobian_nonzero(self):
        """Jacobian should have nonzero entries for all params."""
        for cam_str in TEST_CAMERAS:
            cam = model_selector_from_str(cam_str)
            pts3d = make_pts3d(50).to(cam.device)
            _, valid, J = cam.map_with_jac(pts3d)
            P = cam._data.shape[0]
            for p in range(P):
                col = J[valid, :, p]
                self.assertGreater(col.abs().max().item(), 1e-8,
                                   msg=f'{cam_str}: param {p} has zero Jacobian')

    def test_analytical_vs_fd_float64(self):
        """Analytical Jacobian should match FD in float64."""
        from colmap_cameras.base_model import BaseCamera
        for cam_str in TEST_CAMERAS:
            cam = model_selector_from_str(cam_str)
            cam._data.data = cam._data.data.double()
            pts3d = make_pts3d(20).double()
            _, v1, J_a = cam.map_with_jac(pts3d)
            _, v2, J_f = BaseCamera.map_with_jac(cam, pts3d)
            v = v1 & v2
            if not v.any():
                continue
            abs_diff = (J_a[v] - J_f[v]).abs()
            scale = J_f[v].abs().clamp(min=1e-6)
            rel_err = (abs_diff / scale).max().item()
            self.assertLess(rel_err, 1e-3, msg=f'{cam_str}: rel_err={rel_err}')


class TestUnmapWithJac(unittest.TestCase):

    def test_jacobian_shape(self):
        for cam_str in TEST_CAMERAS_UNMAP:
            cam = model_selector_from_str(cam_str)
            pts2d = make_pts2d(cam, 10)
            pts3d, J = cam.unmap_with_jac(pts2d)
            P = cam._data.shape[0]
            self.assertEqual(J.shape, (10, 3, P), msg=cam_str)

    def test_jacobian_nonzero(self):
        for cam_str in TEST_CAMERAS_UNMAP:
            cam = model_selector_from_str(cam_str)
            pts2d = make_pts2d(cam, 50)
            pts3d, J = cam.unmap_with_jac(pts2d)
            valid = ~torch.isnan(pts3d).any(dim=-1)
            P = cam._data.shape[0]
            for p in range(P):
                col = J[valid, :, p]
                self.assertGreater(col.abs().max().item(), 1e-8,
                                   msg=f'{cam_str}: param {p} has zero Jacobian')


class TestGaussNewtonFit(unittest.TestCase):

    def _test_fit(self, cam_str, perturb_extra=0.05, perturb_focal=0.02):
        cam_gt = model_selector_from_str(cam_str)
        pts3d = make_pts3d(200)

        with torch.no_grad():
            pts2d_gt, valid = cam_gt.map(pts3d)
        pts3d, pts2d_gt = pts3d[valid], pts2d_gt[valid]

        # Perturb
        cam_fit = cam_gt.clone()
        with torch.no_grad():
            fe = cam_fit.num_focal_params
            pe = fe + cam_fit.num_pp_params
            cam_fit._data[:fe] *= (1 + perturb_focal)
            cam_fit._data[pe:] += perturb_extra * torch.randn_like(cam_fit._data[pe:])

        # Initial error
        with torch.no_grad():
            pred, v = cam_fit.map(pts3d)
            err_before = (pred[v] - pts2d_gt[v]).abs().max().item()

        cam_fit._initialize_from_nonlinear(pts2d_gt, pts3d, iterations=20)

        with torch.no_grad():
            pred, v = cam_fit.map(pts3d)
            err_after = (pred[v] - pts2d_gt[v]).abs().max().item()

        self.assertLess(err_after, err_before,
                        msg=f'{cam_str}: before={err_before:.2f} after={err_after:.2f}')
        self.assertLess(err_after, 1.0,
                        msg=f'{cam_str}: err_after={err_after:.2f} > 1px')

    def test_simple_radial(self):
        self._test_fit('SIMPLE_RADIAL 200 200 100 100 100 -0.3')

    def test_radial(self):
        self._test_fit('RADIAL 200 200 100 100 100 0.1 -0.02')

    def test_opencv(self):
        self._test_fit('OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002')

    def test_opencv_fisheye(self):
        self._test_fit('OPENCV_FISHEYE 640 480 300 300 320 240 0.1 -0.05 0.01 -0.005')

    def test_fov(self):
        self._test_fit('FOV 640 480 500 500 320 240 0.8', perturb_extra=0.02)

    def test_pinhole(self):
        self._test_fit('PINHOLE 200 200 100 100 100 100', perturb_extra=0.0)


if __name__ == '__main__':
    unittest.main()
