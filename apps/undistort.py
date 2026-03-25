"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""

import os
import shutil
import torch
import argparse
import cv2
import numpy as np

from colmap_cameras import model_selector
from colmap_cameras.models import SimplePinhole
from colmap_cameras.utils.remapper import Remapper

arg_parser = argparse.ArgumentParser("Generate undistorted image from colmap model")
arg_parser.add_argument("--input_camera", type=str, required=True,
                        help="Input camera model in colmap format. Example: 'OPENCV_FISHEYE 640 480 500 500 320 240 0.1 -0.05 0.01 -0.005'")
arg_parser.add_argument("--fov", type=float, default=120, help="FOV of output pinhole camera in degrees")
arg_parser.add_argument("--output_size", type=int, nargs='+', default=None,
                        help="Output width height. If one number is given it is used for both. Defaults to input size.")
arg_parser.add_argument("--img_path", type=str, default=None, help="Path to input image")
arg_parser.add_argument("--save_lut", type=str, default=None,
                        help="Directory to save LUT (lut.npz + lut_remapper.py)")
arg_parser.add_argument("--output", type=str, default=None, help="Path to save undistorted image")

args = arg_parser.parse_args()

camera_name = args.input_camera.split()[0]
camera_data = list(map(float, args.input_camera.split()[1:]))
input_camera = model_selector(camera_name, camera_data)

# Build output pinhole camera
if args.output_size is not None:
    if len(args.output_size) == 1:
        out_shape = torch.tensor([args.output_size[0], args.output_size[0]], dtype=torch.float32)
    else:
        out_shape = torch.tensor(args.output_size[:2], dtype=torch.float32)
else:
    out_shape = input_camera.image_shape.clone()

output_camera = SimplePinhole.from_fov(args.fov, out_shape)

remapper = Remapper()
xlut, ylut = remapper.compute_lut(input_camera, output_camera)

# Save LUT if requested
if args.save_lut is not None:
    os.makedirs(args.save_lut, exist_ok=True)
    input_w, input_h = [int(x) for x in input_camera.image_shape.tolist()]
    lut_path = os.path.join(args.save_lut, 'lut.npz')
    np.savez(lut_path, xlut=xlut, ylut=ylut, input_size=np.array([input_w, input_h]))
    # Copy standalone remapper script
    src = os.path.join(os.path.dirname(__file__), '..', 'colmap_cameras', 'utils', 'lut_remapper.py')
    shutil.copy2(src, args.save_lut)
    print(f"LUT saved to {lut_path}")

# Remap image if provided
if args.img_path is not None:
    img = cv2.imread(args.img_path)
    w_i, h_i = [int(x.item()) for x in output_camera.image_shape]
    img = cv2.remap(img, xlut, ylut, cv2.INTER_LINEAR)
    img = cv2.resize(img, (w_i, h_i))

    if args.output is not None:
        cv2.imwrite(args.output, img)
        print(f"Saved to {args.output}")
    else:
        cv2.imshow("undistorted", img)
        cv2.waitKey(0)
