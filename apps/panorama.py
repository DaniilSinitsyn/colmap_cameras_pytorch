from colmap_cameras import model_selector_from_str, ValidatedCamera, RotatedCamera, MaskedCamera, CameraRig
from colmap_cameras.models import Equirectangular, SimplePinhole
from colmap_cameras.utils.rig_remapper import RigRemapper
from colmap_cameras.utils.quaternion import quat_to_rotmat
import torch
import cv2
import os
import math
import numpy as np
import argparse


CUBEMAP_FACES = {
    'front':  torch.eye(3),
    'back':   torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]),
    'right':  torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
    'left':   torch.tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
    'up':     torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
    'down':   torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]),
}


def rotation_y(degrees):
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def make_cubemap_face(face_size, R_face, device='cuda'):
    f = face_size / 2.0
    shape = torch.tensor([face_size, face_size])
    params = torch.tensor([f, (face_size - 1) / 2.0, (face_size - 1) / 2.0])
    return RotatedCamera(SimplePinhole(params, shape).to(device), R_face.to(device))


def compose_cubemap_cross(faces, size):
    cross = np.zeros((3 * size, 4 * size, 3), dtype=np.uint8)
    cross[0:size, size:2*size] = faces['up']
    cross[size:2*size, 0:size] = faces['left']
    cross[size:2*size, size:2*size] = faces['front']
    cross[size:2*size, 2*size:3*size] = faces['right']
    cross[size:2*size, 3*size:4*size] = faces['back']
    cross[2*size:3*size, size:2*size] = faces['down']
    return cross


def main():
    parser = argparse.ArgumentParser(description='Stitch front+back fisheye into panorama/cubemap')
    parser.add_argument('front_image', help='Path to front image')
    parser.add_argument('back_image', help='Path to back image')
    parser.add_argument('--front-camera', required=True, help='COLMAP camera string for front camera')
    parser.add_argument('--back-camera', required=True, help='COLMAP camera string for back camera')
    parser.add_argument('--pose', required=True,
                        help='Back camera R_cam2world as quaternion "qw qx qy qz"')
    parser.add_argument('--border', type=int, default=50, help='Source mask border in pixels')
    parser.add_argument('-o', '--output', required=True, help='Output panorama path')
    parser.add_argument('--output-mask', default=None, help='Output mask path')
    parser.add_argument('--cubemap-dir', default=None, help='Output directory for cubemap faces')
    parser.add_argument('--cubemap-size', type=int, default=1024, help='Cubemap face size in pixels')
    parser.add_argument('--cubemap-rotation', type=float, default=0.0,
                        help='Rotate cubemap around Y axis (degrees)')
    args = parser.parse_args()

    quat = torch.tensor([float(x) for x in args.pose.split()])
    R_cam2world = quat_to_rotmat(quat)

    front_model = MaskedCamera(
        ValidatedCamera(model_selector_from_str(args.front_camera).to('cuda')),
        border_size=args.border,
    )
    back_model = MaskedCamera(
        ValidatedCamera(RotatedCamera(
            model_selector_from_str(args.back_camera).to('cuda'),
            R_cam2world.to('cuda'),
        )),
        border_size=args.border,
    )

    rig = CameraRig({'front': front_model, 'back': back_model})
    remapper = RigRemapper(rig, mask_border=10)

    img_front = cv2.imread(args.front_image)
    img_back = cv2.imread(args.back_image)
    assert img_front is not None, f"Cannot read {args.front_image}"
    assert img_back is not None, f"Cannot read {args.back_image}"
    images = {'front': img_front, 'back': img_back}

    # Equirectangular panorama
    w = int(model_selector_from_str(args.front_camera).image_shape[0].item())
    panorama, mask = remapper.remap(
        images, Equirectangular.default_initialization(torch.tensor([2 * w, w])).to('cuda'),
    )
    cv2.imwrite(args.output, panorama)
    print(f"Saved panorama to {args.output}")

    if args.output_mask:
        cv2.imwrite(args.output_mask, mask)
        print(f"Saved mask to {args.output_mask}")

    # Cubemap
    if args.cubemap_dir:
        os.makedirs(args.cubemap_dir, exist_ok=True)
        Ry = rotation_y(args.cubemap_rotation)

        face_images = {}
        face_masks = {}
        for face_name, R_face in CUBEMAP_FACES.items():
            target = make_cubemap_face(args.cubemap_size, Ry @ R_face)
            face_img, face_mask = remapper.remap(images, target)
            face_images[face_name] = face_img
            face_masks[face_name] = face_mask
            cv2.imwrite(os.path.join(args.cubemap_dir, f'{face_name}.jpg'), face_img)
            cv2.imwrite(os.path.join(args.cubemap_dir, f'{face_name}_mask.png'), face_mask)
            print(f"Saved cubemap face: {face_name}")

        cross = compose_cubemap_cross(face_images, args.cubemap_size)
        face_masks_3ch = {k: cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for k, v in face_masks.items()}
        cross_mask = compose_cubemap_cross(face_masks_3ch, args.cubemap_size)
        cross_mask = cv2.cvtColor(cross_mask, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(args.cubemap_dir, 'cross.jpg'), cross)
        cv2.imwrite(os.path.join(args.cubemap_dir, 'cross_mask.png'), cross_mask)
        print("Saved cubemap cross layout")

        # Remap cubemap faces back to fisheye images
        # Pass forward-pass masks so black regions are excluded from blending
        cubemap_cameras = {}
        for name, R_face in CUBEMAP_FACES.items():
            cam = make_cubemap_face(args.cubemap_size, Ry @ R_face)
            cubemap_cameras[name] = MaskedCamera(cam, border_size=0, mask=face_masks[name])
        cubemap_rig = CameraRig(cubemap_cameras)
        cubemap_remapper = RigRemapper(cubemap_rig)

        fisheye_front, _ = cubemap_remapper.remap(face_images, front_model)
        fisheye_back, _ = cubemap_remapper.remap(face_images, back_model)
        fisheye_front[front_model.mask == 0] = 0
        fisheye_back[back_model.mask == 0] = 0

        # Resize originals to match remapped size
        fh, fw = fisheye_front.shape[:2]
        bh, bw = fisheye_back.shape[:2]
        orig_front = cv2.resize(img_front, (fw, fh))
        orig_back = cv2.resize(img_back, (bw, bh))

        top_row = np.hstack([fisheye_front, fisheye_back])
        bottom_row = np.hstack([orig_front, orig_back])
        stacked = np.vstack([top_row, bottom_row])
        cv2.imwrite(os.path.join(args.cubemap_dir, 'fisheye_stacked.jpg'), stacked)
        print("Saved fisheye stacked image (top: remapped, bottom: original)")


if __name__ == '__main__':
    main()
