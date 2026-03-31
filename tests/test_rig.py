"""
2026 Daniil Sinitsyn

Tests for CameraRig and quaternion utils.
"""
import unittest
import math
import torch
from colmap_cameras import model_selector_from_str, CameraRig
from colmap_cameras.utils.quaternion import quat_to_rotmat, rotmat_to_quat


def make_camera():
    return model_selector_from_str('SIMPLE_PINHOLE 200 200 100 100 100')


class TestCameraRig(unittest.TestCase):

    def test_map_returns_dict(self):
        cam1 = make_camera()
        cam2 = make_camera()
        rig = CameraRig({'a': cam1, 'b': cam2})
        pts3d = torch.tensor([[0.0, 0.0, 1.0]])
        results = rig.map(pts3d)
        self.assertIn('a', results)
        self.assertIn('b', results)
        self.assertEqual(len(results), 2)

    def test_map_values(self):
        cam = make_camera()
        rig = CameraRig({'cam': cam})
        pts3d = torch.tensor([[0.0, 0.0, 1.0]])
        result = rig.map(pts3d)
        pts2d, valid = result['cam']
        self.assertTrue(valid.all())

    def test_getitem(self):
        cam = make_camera()
        rig = CameraRig({'cam': cam})
        self.assertIs(rig['cam'], cam)


class TestQuaternion(unittest.TestCase):

    def test_identity(self):
        q = torch.tensor([1., 0., 0., 0.])
        R = quat_to_rotmat(q)
        self.assertLess((R - torch.eye(3)).abs().max().item(), 1e-6)

    def test_90_around_y(self):
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle / 2), 0., math.sin(angle / 2), 0.])
        R = quat_to_rotmat(q)
        expected = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        self.assertLess((R - expected).abs().max().item(), 1e-6)

    def test_180_around_y(self):
        q = torch.tensor([0., 0., 1., 0.])
        R = quat_to_rotmat(q)
        expected = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self.assertLess((R - expected).abs().max().item(), 1e-6)

    def test_roundtrip_quat_to_rotmat(self):
        q = torch.tensor([0.5, 0.5, 0.5, 0.5])
        R = quat_to_rotmat(q)
        q_back = rotmat_to_quat(R)
        R_back = quat_to_rotmat(q_back)
        self.assertLess((R - R_back).abs().max().item(), 1e-5)

    def test_roundtrip_rotmat_to_quat(self):
        axis = torch.tensor([1., 2., 3.])
        axis = axis / axis.norm()
        angle = 1.23
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        R = torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        q = rotmat_to_quat(R)
        R_back = quat_to_rotmat(q)
        self.assertLess((R - R_back).abs().max().item(), 1e-5)

    def test_batch(self):
        q = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
        R = quat_to_rotmat(q)
        self.assertEqual(R.shape, (2, 3, 3))
        q_back = rotmat_to_quat(R)
        self.assertEqual(q_back.shape, (2, 4))

    def test_positive_qw(self):
        R = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        q = rotmat_to_quat(R)
        self.assertGreaterEqual(q[0].item(), 0)


if __name__ == '__main__':
    unittest.main()
