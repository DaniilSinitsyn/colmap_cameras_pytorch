"""
2026 Daniil Sinitsyn

Tests for camera adapters.
"""
import unittest
import math
import numpy as np
import torch
from colmap_cameras import (
    model_selector_from_str, ResizedCamera, ValidatedCamera,
    CompositeCamera, LUTCamera, RotatedCamera, MaskedCamera,
)
from colmap_cameras.adapters.utils import has_adapter


def make_camera():
    return model_selector_from_str('SIMPLE_RADIAL 200 200 100 100 100 -0.3')


def make_camera_strong():
    return model_selector_from_str('SIMPLE_RADIAL 200 200 100 100 100 -2.0')


def make_fisheye():
    return model_selector_from_str('OPENCV_FISHEYE 200 200 100 100 100 100 0.01 -0.01 0.001 -0.001')


class TestResizedCamera(unittest.TestCase):

    def test_image_shape(self):
        cam = ResizedCamera(make_camera(), 0.5)
        self.assertEqual(cam.image_shape.tolist(), [100.0, 100.0])

    def test_image_shape_double(self):
        cam = ResizedCamera(make_camera(), 2.0)
        self.assertEqual(cam.image_shape.tolist(), [400.0, 400.0])

    def test_roundtrip(self):
        inner = make_camera()
        cam = ResizedCamera(inner, 0.5)
        pts2d = torch.tensor([[45., 45.], [50., 50.], [55., 48.]])
        rays = cam.unmap(pts2d)
        pts2d_back, valid = cam.map(rays)
        self.assertTrue(valid.all())
        self.assertLess((pts2d_back - pts2d).abs().max().item(), 1e-2)

    def test_consistency_with_inner(self):
        """Resized unmap at (50,50) with scale=0.5 should match inner unmap at (100,100)."""
        inner = make_camera()
        cam = ResizedCamera(inner, 0.5)
        ray_inner = inner.unmap(torch.tensor([[100., 100.]]))
        ray_resized = cam.unmap(torch.tensor([[50., 50.]]))
        self.assertLess((ray_inner - ray_resized).abs().max().item(), 1e-6)

    def test_get_center(self):
        inner = make_camera()
        cam = ResizedCamera(inner, 0.5)
        c_inner = inner.get_center()
        c_resized = cam.get_center()
        self.assertLess((c_resized - c_inner * 0.5).abs().max().item(), 1e-6)

    def test_compose_with_resize(self):
        """Two resizes should compose: 0.5 * 2.0 = identity."""
        inner = make_camera()
        cam = ResizedCamera(ResizedCamera(inner, 0.5), 2.0)
        pts2d = torch.tensor([[100., 100.]])
        ray_direct = inner.unmap(pts2d)
        ray_composed = cam.unmap(pts2d)
        self.assertLess((ray_direct - ray_composed).abs().max().item(), 1e-6)


class TestValidatedCamera(unittest.TestCase):

    def test_valid_center(self):
        cam = ValidatedCamera(make_camera_strong(), step=2)
        rays, valid = cam.unmap(torch.tensor([[100., 100.]]))
        self.assertTrue(valid[0].item())
        self.assertFalse(torch.isnan(rays).any())

    def test_invalid_edge(self):
        cam = ValidatedCamera(make_camera_strong(), step=2)
        rays, valid = cam.unmap(torch.tensor([[190., 100.]]))
        self.assertFalse(valid[0].item())
        self.assertEqual(rays[0].tolist(), [0.0, 0.0, 0.0])

    def test_map_rejects_invalid_rays(self):
        cam = ValidatedCamera(make_camera_strong(), step=2)
        pts3d = torch.tensor([[3.0, 0.0, 1.0]])  # extreme angle
        pts2d, valid = cam.map(pts3d)
        self.assertFalse(valid[0].item())

    def test_no_nan_in_output(self):
        cam = ValidatedCamera(make_camera_strong(), step=2)
        pts2d = torch.rand(50, 2) * 200
        rays, valid = cam.unmap(pts2d)
        self.assertFalse(torch.isnan(rays).any())

    def test_valid_model_all_valid(self):
        """A well-behaved camera should have most pixels valid."""
        cam = ValidatedCamera(make_camera(), step=2)
        rays, valid = cam.unmap(torch.tensor([[100., 100.], [120., 120.]]))
        self.assertTrue(valid.all())


class TestCompositeCamera(unittest.TestCase):

    def test_inside_boundary_matches_inner(self):
        inner = make_camera_strong()
        cam = CompositeCamera(inner)
        cam.update_boundary()
        pts2d = torch.tensor([[100., 100.]])  # center
        ray_inner = inner.unmap(pts2d)
        ray_comp = cam.unmap(pts2d)
        self.assertLess((ray_inner - ray_comp).abs().max().item(), 1e-6)

    def test_continuation_monotonic(self):
        """Ray angle should increase monotonically with pixel radius."""
        inner = make_camera_strong()
        cam = CompositeCamera(inner)
        cam.update_boundary()
        center = inner.get_center().detach()
        radii = torch.linspace(5, 90, 20)
        pts2d = center + radii[:, None] * torch.tensor([[1., 0.]])
        with torch.no_grad():
            rays = cam.unmap(pts2d)
        rays_n = rays / rays.norm(dim=-1, keepdim=True)
        theta = torch.acos(rays_n[:, 2].clamp(-1, 1))
        diffs = torch.diff(theta)
        self.assertTrue((diffs > -1e-6).all(), f"Non-monotonic: {diffs.tolist()}")

    def test_continuation_bounded(self):
        """Continuation should never exceed pi."""
        inner = make_camera_strong()
        cam = CompositeCamera(inner)
        cam.update_boundary()
        center = inner.get_center().detach()
        pts2d = center + torch.tensor([[150., 0.]])
        with torch.no_grad():
            rays = cam.unmap(pts2d)
        rays_n = rays / rays.norm(dim=-1, keepdim=True)
        import math
        theta = torch.acos(rays_n[0, 2].clamp(-1, 1)).item()
        self.assertLess(theta, math.pi)

    def test_gradients_flow(self):
        """Gradients should reach inner camera parameters through continuation."""
        inner = make_camera_strong()
        cam = CompositeCamera(inner)
        cam.update_boundary()
        inner.requires_grad = True
        center = inner.get_center().detach()
        pts2d = center + torch.tensor([[60., 0.]])  # past boundary
        rays = cam.unmap(pts2d)
        rays.sum().backward()
        self.assertIsNotNone(inner._data.grad)
        self.assertGreater(inner._data.grad.abs().sum().item(), 0)

    def test_monotonicity_loss(self):
        inner = make_camera_strong()
        cam = CompositeCamera(inner)
        cam.update_boundary()
        inner.requires_grad = True
        loss = cam.monotonicity_loss(n_samples=100)
        loss.backward()
        self.assertIsNotNone(inner._data.grad)


class TestAdapterComposition(unittest.TestCase):

    def test_resize_then_validate(self):
        inner = make_camera_strong()
        cam = ValidatedCamera(ResizedCamera(inner, 0.5), step=1)
        rays, valid = cam.unmap(torch.tensor([[50., 50.]]))
        self.assertTrue(valid[0].item())

    def test_validate_then_resize(self):
        inner = make_camera_strong()
        cam = ResizedCamera(ValidatedCamera(inner, step=2), 0.5)
        rays, valid = cam.unmap(torch.tensor([[50., 50.]]))
        self.assertTrue(valid[0].item())

    def test_composite_then_resize(self):
        inner = make_camera_strong()
        comp = CompositeCamera(inner)
        comp.update_boundary()
        cam = ResizedCamera(comp, 0.5)
        rays = cam.unmap(torch.tensor([[50., 50.]]))
        self.assertFalse(torch.isnan(rays).any())


class TestLUTCamera(unittest.TestCase):

    def test_unmap_accuracy(self):
        inner = make_camera()
        lut = LUTCamera(inner, pixel_step=1, angle_step=1.0)
        pts2d = torch.tensor([[100., 100.], [120., 110.], [80., 90.]])
        rays_inner = inner.unmap(pts2d)
        rays_lut = lut.unmap(pts2d)
        self.assertLess((rays_inner - rays_lut).abs().max().item(), 0.01)

    def test_map_accuracy(self):
        inner = make_camera()
        lut = LUTCamera(inner, pixel_step=1, angle_step=0.5)
        pts3d = torch.tensor([[0.1, 0.05, 1.0], [0.0, 0.0, 1.0], [-0.1, 0.1, 1.0]])
        p_inner, v_inner = inner.map(pts3d)
        p_lut, v_lut = lut.map(pts3d)
        v = v_inner & v_lut
        self.assertLess((p_inner[v] - p_lut[v]).abs().max().item(), 1.0)

    def test_unmap_no_nan(self):
        inner = make_camera()
        lut = LUTCamera(inner)
        pts2d = torch.rand(100, 2) * torch.tensor([200., 200.])
        rays = lut.unmap(pts2d)
        self.assertFalse(torch.isnan(rays).any())

    def test_map_valid(self):
        inner = make_camera()
        lut = LUTCamera(inner, angle_step=1.0)
        pts3d = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        pts2d, valid = lut.map(pts3d)
        self.assertTrue(valid.all())

    def test_roundtrip(self):
        inner = make_camera()
        lut = LUTCamera(inner, pixel_step=1, angle_step=0.5)
        pts2d = torch.tensor([[100., 100.], [110., 105.]])
        rays = lut.unmap(pts2d)
        pts2d_back, valid = lut.map(rays)
        self.assertTrue(valid.all())
        self.assertLess((pts2d_back - pts2d).abs().max().item(), 2.0)

    def test_compose_with_resize(self):
        inner = make_camera()
        cam = ResizedCamera(LUTCamera(inner), 0.5)
        rays = cam.unmap(torch.tensor([[50., 50.]]))
        self.assertFalse(torch.isnan(rays).any())


class TestRotatedCamera(unittest.TestCase):

    def test_identity_rotation(self):
        inner = make_camera()
        cam = RotatedCamera(inner, torch.eye(3))
        pts2d = torch.tensor([[100., 100.], [120., 110.]])
        rays_inner = inner.unmap(pts2d)
        rays_rotated = cam.unmap(pts2d)
        self.assertLess((rays_inner - rays_rotated).abs().max().item(), 1e-6)

    def test_identity_map(self):
        inner = make_camera()
        cam = RotatedCamera(inner, torch.eye(3))
        pts3d = torch.tensor([[0.1, 0.05, 1.0]])
        p_inner, v_inner = inner.map(pts3d)
        p_rot, v_rot = cam.map(pts3d)
        self.assertTrue(v_inner.all() and v_rot.all())
        self.assertLess((p_inner - p_rot).abs().max().item(), 1e-6)

    def test_180_rotation(self):
        inner = make_camera()
        R = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        cam = RotatedCamera(inner, R)
        pts2d = torch.tensor([[100., 100.]])
        rays_inner = inner.unmap(pts2d)
        rays_rotated = cam.unmap(pts2d)
        expected = rays_inner.clone()
        expected[:, 0] *= -1
        expected[:, 2] *= -1
        self.assertLess((rays_rotated - expected).abs().max().item(), 1e-5)

    def test_roundtrip(self):
        inner = make_camera()
        cam = RotatedCamera(inner, torch.eye(3))
        pts2d = torch.tensor([[110., 105.]])
        rays = cam.unmap(pts2d)
        pts2d_back, valid = cam.map(rays)
        self.assertTrue(valid.all())
        self.assertLess((pts2d_back - pts2d).abs().max().item(), 1e-2)

    def test_invalid_R_shape(self):
        with self.assertRaises(ValueError):
            RotatedCamera(make_camera(), torch.eye(4))

    def test_model_name(self):
        cam = RotatedCamera(make_camera(), torch.eye(3))
        self.assertIn('Rotated', cam.model_name)

    def test_unmap_tuple_passthrough(self):
        inner = ValidatedCamera(make_fisheye(), step=2)
        cam = RotatedCamera(inner, torch.eye(3))
        result = cam.unmap(torch.tensor([[100., 100.]]))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestMaskedCamera(unittest.TestCase):

    def test_mask_shape(self):
        inner = make_camera()
        cam = MaskedCamera(inner, border_size=10)
        w, h = int(inner.image_shape[0].item()), int(inner.image_shape[1].item())
        self.assertEqual(cam.mask.shape, (h, w))

    def test_mask_dtype(self):
        cam = MaskedCamera(make_camera(), border_size=10)
        self.assertEqual(cam.mask.dtype, np.uint8)

    def test_border_zeros(self):
        cam = MaskedCamera(make_camera(), border_size=10)
        self.assertEqual(cam.mask[0, :].max(), 0)
        self.assertEqual(cam.mask[-1, :].max(), 0)
        self.assertEqual(cam.mask[:, 0].max(), 0)
        self.assertEqual(cam.mask[:, -1].max(), 0)

    def test_center_valid(self):
        cam = MaskedCamera(make_camera(), border_size=10)
        self.assertEqual(cam.mask[100, 100], 255)

    def test_check_bounds_excludes_border(self):
        cam = MaskedCamera(make_camera(), border_size=20)
        pts = torch.tensor([[10., 10.], [100., 100.], [195., 195.]])
        ok = cam.check_bounds(pts)
        self.assertFalse(ok[0].item())
        self.assertTrue(ok[1].item())
        self.assertFalse(ok[2].item())

    def test_map_respects_border(self):
        cam = MaskedCamera(make_camera(), border_size=20)
        # Center should be valid, extreme angle should be invalid (near edge)
        pts3d_center = torch.tensor([[0.0, 0.0, 1.0]])
        pts3d_edge = torch.tensor([[2.0, 2.0, 1.0]])
        _, valid_center = cam.map(pts3d_center)
        _, valid_edge = cam.map(pts3d_edge)
        self.assertTrue(valid_center[0].item())
        self.assertFalse(valid_edge[0].item())

    def test_custom_mask(self):
        inner = make_camera()
        w, h = int(inner.image_shape[0].item()), int(inner.image_shape[1].item())
        custom = np.ones((h, w), dtype=np.uint8) * 255
        custom[:50, :] = 0
        cam = MaskedCamera(inner, border_size=0, mask=custom)
        self.assertEqual(cam.mask[25, 100], 0)
        self.assertEqual(cam.mask[100, 100], 255)

    def test_negative_border_raises(self):
        with self.assertRaises(ValueError):
            MaskedCamera(make_camera(), border_size=-1)

    def test_model_name(self):
        cam = MaskedCamera(make_camera(), border_size=10)
        self.assertIn('Masked', cam.model_name)


class TestHasAdapter(unittest.TestCase):

    def test_direct(self):
        cam = MaskedCamera(make_camera(), border_size=10)
        self.assertTrue(has_adapter(cam, MaskedCamera))

    def test_nested(self):
        cam = MaskedCamera(ValidatedCamera(make_camera(), step=2), border_size=10)
        self.assertTrue(has_adapter(cam, ValidatedCamera))
        self.assertTrue(has_adapter(cam, MaskedCamera))

    def test_missing(self):
        cam = ValidatedCamera(make_camera(), step=2)
        self.assertFalse(has_adapter(cam, MaskedCamera))

    def test_raw_camera(self):
        self.assertFalse(has_adapter(make_camera(), MaskedCamera))


if __name__ == '__main__':
    unittest.main()
