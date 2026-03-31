# :globe_with_meridians: Colmap Cameras in PyTorch

[![CI](https://github.com/DaniilSinitsyn/colmap_cameras_pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/DaniilSinitsyn/colmap_cameras_pytorch/actions/workflows/ci.yml)

PyTorch implementations of camera models from [COLMAP](https://colmap.github.io/) and beyond.

Camera models are `torch.nn.Module` subclasses with full **automatic differentiation** for `map` (project) and `unmap` (backproject) operations.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Camera Models](#camera-models)
- [Camera Wrappers](#camera-wrappers)
- [Camera Rig](#camera-rig)
- [Valid Region Estimation](#valid-region-estimation)
- [Apps](#apps)
- [Root Solvers](#root-solvers)
- [Analytical Jacobians](#analytical-jacobians)
- [Tests](#tests)

## Installation

```bash
pip install git+https://github.com/DaniilSinitsyn/colmap_cameras_pytorch.git
```

Or clone directly:

```bash
git clone https://github.com/DaniilSinitsyn/colmap_cameras_pytorch.git
```

## Basic Usage

### Load camera from cameras.txt

```python
import torch
from colmap_cameras import model_selector, model_selector_from_str

# From a colmap string
model = model_selector_from_str("SIMPLE_RADIAL 100 100 100 50 50 0.3")

# Or from name + parameters
model = model_selector("SIMPLE_RADIAL", [100, 100, 100, 50, 50, 0.3])

# Project 3D points to pixels
pts2d, valid = model.map(pts3d)

# Backproject pixels to rays
rays = model.unmap(pts2d)
```

### Optimizing camera parameters

```python
model.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(iterations):
    optimizer.zero_grad()
    pts2d, valid = model.map(pts3d)
    loss = ((pts2d[valid] - target[valid]) ** 2).mean()
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

### Rescaling a camera

```python
half_res = model.rescale(0.5)   # new standalone camera with halved focal, center, image size
print(half_res.to_colmap())     # exportable
```

## Camera Models

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
| MEIS_EXTENDED_CAMERA_MODEL| `MeisExtendedCameraModel`| Mei model with k3 radial distortion  |
| DIVISION_MODEL           | `DivisionModel`           | Single-parameter division undistortion|
| POLYNOMIAL_DIVISION_MODEL| `PolynomialDivisionModel` | Multi-parameter division model       |
| WOODSCAPE               | `WoodScape`                | Valeo WoodScape fisheye model        |

## Camera Wrappers

Camera wrappers live in `colmap_cameras.adapters` and inherit from `CameraAdapter`, which delegates all common methods to the inner model. They compose freely in any order.

| Wrapper | Purpose | `map()` | `unmap()` |
|:-------:|:--------|:--------|:----------|
| `ValidatedCamera` | Filters invalid points | Checks ray against spherical validity map | Returns `(rays, valid)`, zeros for invalid |
| `CompositeCamera` | Extends to full sphere | Delegates to inner | Atan continuation past valid boundary |
| `ResizedCamera` | Scale resolution | Scales output pixels | Scales input pixels |
| `LUTCamera` | Precomputed fast lookup | Spherical $(\theta, \phi)$ LUT | Pixel grid LUT |
| `RotatedCamera` | Extrinsic rotation | Rotates points world→cam | Rotates rays cam→world |
| `MaskedCamera` | Border + validity mask | Excludes border pixels | Delegates to inner |

**`ValidatedCamera`** — filters out points outside the valid region. Precomputes a lookup table in spherical coordinates $(\theta, \phi)$ so validity checks are $O(1)$ per ray. Both `map()` and `unmap()` return `(result, valid)` tuples with zero vectors for invalid points (no NaN).

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

**`ResizedCamera`** — scales pixel coordinates by a factor. Useful for working at reduced resolution.

```python
from colmap_cameras import ResizedCamera

half = ResizedCamera(inner_camera, scale=0.5)   # half resolution
print(half.to_colmap())                          # exports rescaled params
```

**`LUTCamera`** — precomputes `map` and `unmap` into dense lookup tables. Uses bilinear interpolation via `grid_sample` for fast evaluation. `unmap` uses a pixel grid LUT, `map` uses a spherical $(\theta, \phi)$ grid LUT.

```python
from colmap_cameras import LUTCamera

cam = LUTCamera(inner_camera, pixel_step=1, angle_step=0.5)
pts2d, valid = cam.map(pts3d)   # spherical LUT lookup
rays = cam.unmap(pts2d)          # pixel LUT lookup
```

**`RotatedCamera`** — applies an extrinsic rotation to rays. `R_cam2world` transforms rays from the camera's local frame into the world frame. Use this to place cameras in a multi-camera rig.

```python
from colmap_cameras import RotatedCamera

cam = RotatedCamera(inner_camera, R_cam2world)
rays = cam.unmap(pts2d)        # rays in world frame
pts2d, valid = cam.map(pts3d)  # pts3d in world frame
```

**`MaskedCamera`** — shrinks the valid region by a pixel border and builds a uint8 mask image `(h, w)` combining validity with border exclusion. The mask is suitable for `cv2.remap` and saving to disk. A pre-computed mask can be passed to skip validity estimation.

```python
from colmap_cameras import MaskedCamera

cam = MaskedCamera(ValidatedCamera(inner), border_size=50)
cam.mask    # (h, w) uint8 numpy array, 255=valid, 0=invalid

# Or with a pre-computed mask
cam = MaskedCamera(inner, border_size=10, mask=my_mask)
```

Wrappers compose:

```python
cam = ResizedCamera(ValidatedCamera(inner), scale=0.5)
# or
cam = ValidatedCamera(ResizedCamera(inner, 0.5))
```

## Camera Rig

`CameraRig` holds a named collection of cameras and projects world-frame 3D points through all of them in one call.

```python
from colmap_cameras import CameraRig, RotatedCamera, MaskedCamera, ValidatedCamera

rig = CameraRig({
    'front': MaskedCamera(ValidatedCamera(front_model), border_size=50),
    'back':  MaskedCamera(ValidatedCamera(RotatedCamera(back_model, R_cam2world)), border_size=50),
})

results = rig.map(pts3d)   # {'front': (pts2d, valid), 'back': (pts2d, valid)}
rig['front']               # access individual cameras
```

### RigRemapper

`RigRemapper` remaps images from a camera rig into any target camera's pixel space using distance-based blending. Each camera's weight is proportional to its distance from the valid region boundary, producing smooth seams in overlap regions.

```python
from colmap_cameras.utils.rig_remapper import RigRemapper

remapper = RigRemapper(rig, mask_border=10)

# One-off: build LUTs and remap in one call
image, mask = remapper.remap({'front': img_f, 'back': img_b}, target_camera)

# Batch: precompute LUTs once, apply to many frames
luts = remapper.build_luts(target_camera)
for img_f, img_b in frame_pairs:
    image, mask = RigRemapper.remap_from_luts({'front': img_f, 'back': img_b}, luts)
```

The `mask_border` parameter erodes each camera's mask by that many pixels in target space, creating a zero-gap of `2 * mask_border` between adjacent cameras.

Any target camera works — equirectangular for panoramas, pinhole for cubemap faces:

```python
from colmap_cameras.models import Equirectangular, SimplePinhole

# Panorama
target = Equirectangular.default_initialization(torch.tensor([7680, 3840])).to('cuda')

# Cubemap face (90° FoV pinhole, rotated)
pinhole = SimplePinhole(torch.tensor([512.0, 511.5, 511.5]), torch.tensor([1024, 1024]))
target = RotatedCamera(pinhole.to('cuda'), R_face.to('cuda'))
```

## Valid Region Estimation

Camera distortion can become non-monotonic — the mapping $\text{pixel} \to \text{ray}$ develops a fold where nearby pixels map to wildly different ray directions. We detect this by checking the **round-trip Jacobian**.

For a pixel $\mathbf{p}$, consider the composition $g(\mathbf{p}) = \text{map}(\text{unmap}(\mathbf{p}))$. In the valid region this is the identity, so $J_g = I$ and $\det(J_g) = 1$. At a fold, Newton's root finder inside `unmap` converges to a wrong root, so $g(\mathbf{p}) \neq \mathbf{p}$ and $\det(J_g)$ deviates from 1. The criterion is:

$$\left|\det\!\left(\frac{\partial\, \text{map}(\text{unmap}(\mathbf{p}))}{\partial \mathbf{p}}\right) - 1\right| < \varepsilon$$

We compute $\det(J_g)$ via finite differences on the round-trip $g(\mathbf{p}) = \text{map}(\text{unmap}(\mathbf{p}))$, then flood-fill from the principal point to get the largest connected valid component. No autograd needed.

```python
from colmap_cameras.utils.valid_region import estimate_valid_region

valid_mask = estimate_valid_region(camera, step=2)  # (H, W) bool tensor
```

## Apps

Apps can be run from the repo root (`python3 -m apps.<name>`) or after pip install (`python3 -m colmap_cameras.apps.<name>`).

### Panorama / Cubemap

Stitches a front+back fisheye pair into an equirectangular panorama and/or cubemap.

```bash
python3 -m apps.panorama front.jpg back.jpg \
  --front-camera "OPENCV_FISHEYE 3840 3840 1015.8 1015.7 1931.8 1924.4 0.019 0.023 -0.008 0.0002" \
  --back-camera  "OPENCV_FISHEYE 3840 3840 1017.4 1017.5 1930.6 1940.5 0.020 0.022 -0.008 0.0002" \
  --pose "0.003 -0.008 -1.0 -0.002" \
  -o panorama.jpg --output-mask mask.png

# With cubemap output (optionally rotated 45° so faces straddle source cameras)
python3 -m apps.panorama front.jpg back.jpg \
  --front-camera "..." --back-camera "..." --pose "qw qx qy qz" \
  -o panorama.jpg --cubemap-dir ./cubemap --cubemap-rotation 45
```

`--pose` is the back camera's `R_cam2world` as a quaternion `(qw, qx, qy, qz)`. Use `colmap_cameras.utils.quaternion.rotmat_to_quat` to convert from a rotation matrix.

### Refit model

Fits one camera model to another using Gauss-Newton with analytical Jacobians. Principal point is always fixed.

```bash
python3 -m colmap_cameras.apps.refit_model \
  --input_camera "SIMPLE_RADIAL 100 100 100 50 50 0.3" \
  --output_camera "RADIAL_FISHEYE" --iterations 20

python3 -m colmap_cameras.apps.refit_model \
  --input_camera "MEIS_EXTENDED... 3800 3800 ..." \
  --output_camera "OPENCV_FISHEYE" --gpu --resize_ratio 0.5
```

### Undistort

Generates undistorted pinhole images. Can save LUTs for fast remapping without torch.

```bash
python3 -m colmap_cameras.apps.undistort \
  --input_camera "OPENCV_FISHEYE 640 480 500 500 320 240 0.1 -0.05 0.01 -0.005" \
  --fov 120 --output_size 512 \
  --save_lut ./my_lut \
  --img_path input.png --output undistorted.png
```

### Valid region visualization

Interactive visualization with radial profile and azimuth slider.

```bash
python3 -m colmap_cameras.apps.valid_region \
  --input_camera "SIMPLE_RADIAL 200 200 100 100 100 -2.0"
```

### Camera model remapper

```python
from colmap_cameras.utils.remapper import Remapper
remapper = Remapper(step=4)
img = remapper.remap(model_in, model_out, img_path)
img = remapper.remap_from_fov(model_in, fov_out, img_path)  # fov in degrees
```

## Root Solvers

Camera models with polynomial distortion require inverting the distortion function. Set `ROOT_FINDING_METHOD` to choose the solver:

```python
from colmap_cameras.base_model import BaseCamera
BaseCamera.ROOT_FINDING_METHOD = 'newton'     # default, autograd through iterations
BaseCamera.ROOT_FINDING_METHOD = 'companion'  # eigenvalue solver (auto-falls back to Newton if ill-conditioned)
```

Newton's method differentiates through the iterations directly (not via the implicit function theorem), which correctly exposes fold singularities in the gradient.

## Analytical Jacobians

Most camera models provide `map_with_jac(pts3d)` which returns `(pts2d, valid, J)` where `J` is the `(N, 2, P)` Jacobian of the projection w.r.t. all camera parameters, computed analytically. Models without an analytical implementation fall back to finite differences. Used internally by the Gauss-Newton fitter.

## Tests

```bash
python3 -m tests.run_tests -v
```
