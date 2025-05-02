"""
2025 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""

import torch
import argparse
import cv2

from colmap_cameras import model_selector, default_initialization
from colmap_cameras.utils.remapper import Remapper

arg_parser = argparse.ArgumentParser("Generate undistorted image from colmap model")
arg_parser.add_argument("--input_camera", type=str, help="Input camera model in colmap foramt. Example: PINHOLE 100 100 100 100 50 50")
arg_parser.add_argument("--fov", type=float, default=120, help="FOV of output pinhole camera")
arg_parser.add_argument("--img_path", type=str, help="Path to input image")

args = arg_parser.parse_args()
input_camera = args.input_camera
fov_out = args.fov
img_path = args.img_path

input_camera_name = input_camera.split()[0]
input_camera_data = list(map(float, input_camera.split()[1:]))
input_camera = model_selector(input_camera_name, input_camera_data)

remapper = Remapper()
img = remapper.remap_from_fov(input_camera, fov_out, img_path)

cv2.imshow("undistorted.png", img)
cv2.waitKey(0)

