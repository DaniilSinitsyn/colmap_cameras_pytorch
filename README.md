# :globe_with_meridians: Colmap Cameras in PyTorch

This repository contains PyTorch implementations of the camera models used in the [COLMAP](https://colmap.github.io/) structure-from-motion pipeline.

Camera models are `torch.nn.Module` subclasses with full **automatic differentiation** for `map` (project) and `unmap` (backproject) operations.

> This code was mainly developed for my own research purposes.

## Installation

Just `git clone` this repository to your project folder.

```bash
git clone https://github.com/DaniilSinitsyn/colmap_cameras_pytorch.git
```

## Usage

### Load camera from cameras.txt

```python
import torch
from colmap_cameras_pytorch.colmap_cameras import model_selector, model_selector_from_str

# Model defined as a string in colmap cameras.txt
camera_txt = "SIMPLE_RADIAL 100 100 100 50 50 0.3"

# Create model based on the colmap string
model = model_selector_from_str(camera_txt)

# Manually create model from parameters and model name
camera_model = camera_txt.split()[0]
camera_params = torch.tensor([float(x) for x in camera_txt.split()[1:]])

model = model_selector(camera_model, camera_params)

# project 3d points onto image
pts3d = torch.rand(10, 3)
points_2d, valid = model.map(points3d)
# unproject 2d points to the ray
points_3d = model.unmap(points_2d)
```

### Optimizing camera parameters

Camera models are `torch.nn.Module` subclasses. Enable gradients and use any PyTorch optimizer:

```python
model.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(iterations):
    optimizer.zero_grad()
    ...
    loss = ...
    loss.backward()
    optimizer.step()
```

### Fixing parameters

By default the principal point (center) is fixed. Use `fix()` / `unfix()` to control which parameters are optimized:

```python
model.unfix('center')           # unfix principal point
model.fix('focal')              # fix focal length
model.fix(4, 5)                 # fix individual params by index

print(model.fixed)              # {'focal': True, 'center': False, 'extra': False}
```

Available parameter groups: `'focal'`, `'center'`, `'extra'`.


### Camera models

[All camera models are supported](colmap_cameras/models):

| Colmap's name         | PyTorch class        |
| :-------------------: | :------------------:  |
| SIMPLE_PINHOLE        | `SimplePinhole`      |
| PINHOLE               | `Pinhole`            |
| SIMPLE_RADIAL         | `SimpleRadial`       |
| RADIAL                | `Radial`             |
| OPENCV                | `OpenCV`             |
| OPENCV_FISHEYE        | `OpenCVFisheye`      |
| FULL_OPENCV           | `FullOpenCV`         |
| SIMPLE_RADIAL_FISHEYE | `SimpleRadialFisheye`|
| RADIAL_FISHEYE        | `RadialFisheye`      |
| FOV                   | `Fov`                |
| THIN_PRISM_FISHEYE    | `ThinPrismFisheye`   |

To use a specific camera model you can import it directly from the `colmap_cameras.models` module.

```python
import torch
from colmap_cameras_pytorch.colmap_cameras.models import Pinhole

image_shape = torch.tensor([[100, 100]]).float()
params = torch.tensor([100, 100, 50, 50]).float()
model = Pinhole(params, image_shape)
```

## Useful stuff

### Apps

[`apps.refit_model`](apps/refit_model.py) is a simple script that fits one camera model to another.

```bash
python3 -m apps.refit_model --input_camera "SIMPLE_RADIAL 100 100 100 50 50 0.3"  --output_camera "RADIAL_FISHEYE" --iterations 20
```

### Camera model remapper

[`colmap_cameras.utils.remapper`](colmap_cameras/utils/remapper.py) is a class that can be used to remap one camera model to another.

```python
from colmap_cameras_pytorch.colmap_cameras.utils.remapper import Remapper
remapper = Remapper(step = 4) # the step of arange for the image grid
img = remapper.remap(model_in, model_out, img_path)
img = remapper.remap_from_fov(model_in, fov_out, img_path) # fov in degrees
```

### Root solvers

Some camera models require solving polynomial roots. [For high-order polynomials, the only way to do this is to use a numerical solver.](https://en.wikipedia.org/wiki/Abel–Ruffini_theorem)

This repo contains an extention of `torch.autograd.Function` for [Newton's method](colmap_cameras/utils/newton_root_1d.py) and [Companion matrix root solver](colmap_cameras/utils/companion_matrix_root_1d.py).


## Tests

To run tests:

```bash
python3 -m tests.run_tests -v
```

### Generating and using LUTs

The undistort app can save look-up tables for fast remapping without recomputing the projection:

```bash
python3 -m apps.undistort \
  --input_camera "OPENCV_FISHEYE 640 480 500 500 320 240 0.1 -0.05 0.01 -0.005" \
  --fov 120 --output_size 512 \
  --save_lut ./my_lut \
  --img_path input.png --output undistorted.png
```

A standalone `lut_remapper.py` is copied into the output directory. Use it to remap images without any torch dependency:

```python
from lut_remapper import LutRemapper
remapper = LutRemapper('./my_lut/lut.npz')
img = remapper.remap('input.png')  # validates input image size
```

`--output_size` accepts one number (square) or two (width height).

## TODO
- [x] Add remap app, that generates remaps alongside with a class to run them.
- [ ] Estimate image area where camera is valid for each model. (Basically to check whether distortion is monotonic)
- [ ] Visualisation util for the previous point.
