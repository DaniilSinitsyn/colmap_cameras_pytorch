"""
2026 Daniil Sinitsyn

Tests for initialize_from / initialize_distortion_from_points.
"""
import unittest
import torch
from colmap_cameras import model_selector_from_str


TEST_CASES = [
    ('SIMPLE_RADIAL 200 200 100 100 100 -0.3', 'SIMPLE_RADIAL'),
    ('SIMPLE_RADIAL 200 200 100 100 100 -0.3', 'RADIAL'),
    ('OPENCV_FISHEYE 640 480 300 300 320 240 0.1 -0.05 0.01 -0.005', 'OPENCV_FISHEYE'),
    ('OPENCV_FISHEYE 640 480 300 300 320 240 0.1 -0.05 0.01 -0.005', 'SIMPLE_RADIAL_FISHEYE'),
    ('OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002', 'OPENCV'),
    ('OPENCV 640 480 500 500 320 240 0.1 -0.05 0.001 0.002', 'FULL_OPENCV'),
    ('FOV 640 480 500 500 320 240 0.8', 'FOV'),
    ('UNIFIED_CAMERA_MODEL 200 200 100 100 100 100 0.5', 'UNIFIED_CAMERA_MODEL'),
    ('DIVISION_MODEL 200 200 100 100 100 0.1', 'DIVISION_MODEL'),
    ('SIMPLE_RADIAL 200 200 100 100 100 -0.1', 'OPENCV_FISHEYE'),
]


class TestInitializeFrom(unittest.TestCase):

    def _test_refit(self, src_str, dst_model, max_err=10.0):
        from colmap_cameras import default_initialization
        src = model_selector_from_str(src_str)
        dst = default_initialization(dst_model, src.image_shape)

        dst.initialize_from(src, nonlinear_iterations=30)

        # Sample test points and check reprojection
        torch.manual_seed(42)
        pts3d = torch.randn(200, 3)
        pts3d[:, 2] = pts3d[:, 2].abs() + 0.5

        with torch.no_grad():
            p_src, v_src = src.map(pts3d)
            p_dst, v_dst = dst.map(pts3d)
            v = v_src & v_dst
            if v.sum() == 0:
                return
            err = (p_src[v] - p_dst[v]).pow(2).sum(dim=-1).sqrt()

        self.assertLess(err.mean().item(), max_err,
                        msg=f'{src_str} -> {dst_model}: mean_err={err.mean().item():.2f}px')


def _make_test(src_str, dst_model):
    def test(self):
        self._test_refit(src_str, dst_model)
    return test


for src_str, dst_model in TEST_CASES:
    name = f'test_{src_str.split()[0].lower()}_to_{dst_model.lower()}'
    setattr(TestInitializeFrom, name, _make_test(src_str, dst_model))


if __name__ == '__main__':
    unittest.main()
