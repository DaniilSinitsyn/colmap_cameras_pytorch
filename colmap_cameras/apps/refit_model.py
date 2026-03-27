"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import argparse

from colmap_cameras import model_selector, default_initialization, ResizedCamera


def main():
    arg_parser = argparse.ArgumentParser("Refit one colmap model to another")
    arg_parser.add_argument("--input_camera", type=str, help="Input camera model in colmap format")
    arg_parser.add_argument("--output_camera", type=str, help="Output camera model name")
    arg_parser.add_argument("--iterations", type=int, default=20, help="Number of Gauss-Newton iterations")
    arg_parser.add_argument("--gpu", action="store_true", help="Use GPU")
    arg_parser.add_argument("--resize_ratio", type=float, default=None, help="Fit to a rescaled version of the input camera")

    args = arg_parser.parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    input_camera = model_selector(args.input_camera.split()[0],
                                  list(map(float, args.input_camera.split()[1:])), device=device)

    if args.resize_ratio is not None:
        input_camera = ResizedCamera(input_camera, args.resize_ratio)

    output_camera = default_initialization(args.output_camera, input_camera.image_shape, device=device)
    output_camera.initialize_from(input_camera, args.iterations)

    print("Input model : \n\t", input_camera.to_colmap())
    print("Output model: \n\t", output_camera.to_colmap())


if __name__ == '__main__':
    main()
