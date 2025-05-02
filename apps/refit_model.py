"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import argparse

from colmap_cameras import model_selector, default_initialization

arg_parser = argparse.ArgumentParser("Refit one colmap model to another")
arg_parser.add_argument("--input_camera", type=str, help="Input camera model in colmap foramt. Example: PINHOLE 100 100 100 100 50 50")
arg_parser.add_argument("--output_camera", type=str, help="Output camera model in colmap format. Example: SIMPLE_RADIAL")
arg_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for optimization")
arg_parser.add_argument("--gpu", action="store_true", help="Use GPU for optimization")


args = arg_parser.parse_args()
input_camera = args.input_camera
output_camera = args.output_camera

device = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("GPU is not available, using CPU instead")
        device = 'cpu'

input_camera_name = input_camera.split()[0]
input_camera_data = list(map(float, input_camera.split()[1:]))

input_camera = model_selector(input_camera_name, input_camera_data, device=device)
output_camera = default_initialization(output_camera, input_camera.image_shape, device=device)

output_camera.initialize_from(input_camera, args.iterations)

print("Input model : \n\t", input_camera.to_colmap())
print("Output model init : \n\t", output_camera.to_colmap())
