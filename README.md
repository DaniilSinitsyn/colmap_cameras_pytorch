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

**COLMAP models:**

| Name                  | Class                |
| :-------------------: | :------------------: |
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

**Additional models:**

| Name                     | Class                     | Description                          |
| :----------------------: | :-----------------------: | :----------------------------------: |
| EQUIRECTANGULAR          | `Equirectangular`         | Spherical panoramic projection       |
| UNIFIED_CAMERA_MODEL     | `UnifiedCameraModel`      | Geyer-Barreto unified model (UCM)    |
| MEIS_CAMERA_MODEL        | `MeisCameraModel`         | Enhanced unified model (Mei)         |
| DIVISION_MODEL           | `DivisionModel`           | Single-parameter division undistortion|
| POLYNOMIAL_DIVISION_MODEL| `PolynomialDivisionModel` | Multi-parameter division model       |
| WOODSCAPE               | `WoodScape`                | Valeo WoodScape fisheye model        |

To use a specific camera model you can import it directly from the `colmap_cameras.models` module.

```python
import torch
from colmap_cameras_pytorch.colmap_cameras.models import Pinhole

image_shape = torch.tensor([[100, 100]]).float()
params = torch.tensor([100, 100, 50, 50]).float()
model = Pinhole(params, image_shape)
```

### Valid region estimation

Camera distortion can become non-monotonic — the mapping $\text{pixel} \to \text{ray}$ develops a fold where nearby pixels map to wildly different ray directions. We detect this by checking the **round-trip Jacobian**.

For a pixel $\mathbf{p}$, consider the composition $g(\mathbf{p}) = \text{map}(\text{unmap}(\mathbf{p}))$. In the valid region this is the identity, so $J_g = I$ and $\det(J_g) = 1$. At a fold, Newton's root finder inside `unmap` converges to a wrong root, so $g(\mathbf{p}) \neq \mathbf{p}$ and $\det(J_g)$ deviates from 1. The criterion is:

$$\left|\det\!\left(\frac{\partial\, \text{map}(\text{unmap}(\mathbf{p}))}{\partial \mathbf{p}}\right) - 1\right| < \varepsilon$$

We compute $\det(J_g)$ via two backward passes through the full $\text{map}(\text{unmap}(\mathbf{p}))$ chain, then flood-fill from the principal point to get the largest connected valid component.

This requires differentiating through Newton's iterations (not the implicit function theorem), because the IFT backward hides root-switching discontinuities.

```python
from colmap_cameras.utils.valid_region import estimate_valid_region

valid_mask = estimate_valid_region(camera, step=2)  # (H, W) bool tensor
```

Interactive visualization with radial profile and slider:

```bash
python3 -m apps.valid_region --input_camera "SIMPLE_RADIAL 200 200 100 100 100 -2.0"
```

### Camera wrappers

Two wrappers can be applied to any camera model. Both inherit from `CameraAdapter` which delegates all common methods (`map`, `get_center`, `fix`, `to_colmap`, etc.) to the inner model.

**`ValidatedCamera`** — filters out points outside the valid region. Precomputes a lookup table in spherical coordinates `(theta, phi)` so validity checks are O(1) per ray. Both `map()` and `unmap()` return `(result, valid)` tuples with zero vectors for invalid points (no NaN).

```python
from colmap_cameras import ValidatedCamera

cam = ValidatedCamera(inner_camera)
pts2d, valid = cam.map(pts3d)     # valid=False if ray direction is outside valid region
rays, valid = cam.unmap(pts2d)    # zero vectors outside, no NaN
```

**`CompositeCamera`** — extends any camera model to cover the full sphere by replacing the inner model's `unmap()` past the valid boundary with a smooth analytical continuation.

The continuation uses a shifted arctangent:

$$\theta(r) = \theta_b + k \cdot \arctan\!\left(\frac{(r - r_b) \cdot s_b}{k}\right), \quad k = \frac{\pi - \theta_b}{\pi/2}$$

where $r$ is pixel radius from the center, $r_b$ is the boundary radius (from the valid region mask), and $\theta_b$, $s_b = d\theta/dr$ are the ray angle and slope at the boundary, computed **live** from the inner model's `unmap()` so that gradients flow through to the camera parameters.

This gives:
- **C0 continuity**: $\theta(r_b) = \theta_b$
- **C1 continuity**: $\theta'(r_b) = s_b$
- **Monotonicity**: $\theta'(r) > 0$ for all $r > r_b$
- **Boundedness**: $\theta(r) \to \pi$ as $r \to \infty$

Includes a differentiable monotonicity regularizer that penalizes $d\theta/dr$ approaching zero near the boundary — this provides gradient signal *before* a fold happens, unlike the round-trip Jacobian which only detects folds after the fact.

```python
from colmap_cameras import CompositeCamera

cam = CompositeCamera(inner_camera)
cam.update_boundary()

rays = cam.unmap(pixels)                              # smooth past the fold
loss = reprojection + 0.1 * cam.monotonicity_loss()   # prevents folding during optimization
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

Camera models with polynomial distortion require inverting the distortion function. Set `ROOT_FINDING_METHOD` to choose the solver:

```python
from colmap_cameras.base_model import BaseCamera
BaseCamera.ROOT_FINDING_METHOD = 'newton'     # default, autograd through iterations
BaseCamera.ROOT_FINDING_METHOD = 'companion'  # eigenvalue solver (auto-falls back to Newton if ill-conditioned)
```

Newton's method differentiates through the iterations directly (not via the implicit function theorem), which correctly exposes fold singularities in the gradient.


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
