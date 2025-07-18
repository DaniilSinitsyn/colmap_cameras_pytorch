# :globe_with_meridians: Colmap Cameras in PyTorch

This repository contains PyTorch implementations of the camera models used in the [COLMAP](https://colmap.github.io/) structure-from-motion pipeline.

The camera models support **automatic differentiation** for `project` and `backproject` functions. Which for some reason are called `map` and `unmap` in this repo.

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

As everything is differentiable, you can optimize camera parameters using PyTorch's optimizers.

```python
model.require_grad = True
optimizer = torch.optim.Adam([model._data], lr=0.01)

for _ in range(iterations):
    optimizer.zero_grad()
    ...
    loss = ...
    loss.backward()
    optimizer.step()
```

By default camera's center is fixed. If you want to optimize it too:

```python
model.OPTIMIZATION_FIX_CENTER = False
```

There are in total 4 flags that can be set for each camera:

- `OPTIMIZATION_FIX_FOCALS`: Fix focal lengthes (default: `False`)
- `OPTIMIZATION_FIX_CENTER`: Fix principal point (default: `True`)
- `OPTIMIZATION_FIX_EXTRA`: Fix extra parameters (default: `False`)
- `ROOT_FINDING_MAX_ITERATIONS`: Number of iterations for root finding (default: `50`)


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

## Usefu stuff

### Apps

[`apps.refit_model`](apps/refit_model.py) is a simple script that uses Gauss-Newton optimization to fit one camera model to another.

```bash
python3 -m apps.refit_model --input_camera "SIMPLE_RADIAL 100 100 100 50 50 0.3"  --output_camera "RADIAL_FISHEYE" --iterations 20
```

### Camera model remapper

[`colmap_cameras.utils.remapper`](colmap_cameras/utils/remapper.py) is a class that can be used to remap one camera model to another.

```python
from colmap_cameras_pytorch.colmap_cameras.util.remapper import Remapper
remapper = Remapper(step = 4) # the step of arange for the image grid
img = remapper.remap(model_in, model_out, img_path) 
img = remapper.remap_from_fov(model_in, fov_out, img_path) # fov in degrees
```

### Root solvers

Some camera models require solving polynomial roots. [For high-order polynomials, the only way to do this is to use a numerical solver.](https://en.wikipedia.org/wiki/Abel–Ruffini_theorem)

>I don't like the fact that automatic differentiation goes through Newton's method or the QR algorithm.

This repo contains an extention of `torch.autograd.Function` for [Newton's method](colmap_cameras/utils/newton_root_1d.py) and [Companion matrix root solver](colmap_cameras/utils/companion_matrix_root_1d.py).


## Tests

To run tests:

```bash
python3 -m tests.run_tests -v
```

## TODO
- [ ] Add remap app, that generates remaps alongside with a class to run them.
- [ ] Estimate image area where camera is valid for each model. (Basically to check whether distortion is monotonic)
- [ ] Visualisation util for the previous point.




