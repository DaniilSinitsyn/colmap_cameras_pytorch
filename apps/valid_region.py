"""
2026 Daniil Sinitsyn

Interactive visualization of camera valid region and radial distortion profile.

Left panel: image with valid region overlay and current radial cut line.
Right panel: pixel radius vs ray angle along the current cut direction.
Slider controls the azimuthal angle of the cut.
"""
import os
import sys
import argparse
import math

# Pick a non-Qt matplotlib backend to avoid conflicts with OpenCV's Qt.
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('macosx')
elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from colmap_cameras import model_selector_from_str
from colmap_cameras.utils.valid_region import estimate_valid_region


def compute_radial_profile(camera, azimuth, n_samples=500):
    """Compute pixel radius vs ray angle along one azimuthal direction."""
    device = camera.device
    center = camera.get_center().detach()
    if center.numel() < 2:
        w, h = [int(x.item()) for x in camera.image_shape]
        center = torch.tensor([w / 2.0, h / 2.0], device=device)

    max_r = max(camera.image_shape[0].item(), camera.image_shape[1].item())
    radii = torch.linspace(0, max_r, n_samples, device=device)

    dx = torch.cos(torch.tensor(azimuth, device=device))
    dy = torch.sin(torch.tensor(azimuth, device=device))
    pts2d = center.unsqueeze(0) + radii.unsqueeze(1) * torch.stack([dx, dy]).unsqueeze(0)

    with torch.no_grad():
        rays = camera.unmap(pts2d)

    valid = ~torch.isnan(rays).any(dim=-1)
    rays_norm = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_angle = rays_norm[:, 2].clamp(-1, 1)
    ray_angle = torch.acos(cos_angle) * 180 / math.pi

    return radii.cpu().numpy(), ray_angle.cpu().numpy(), valid.cpu().numpy()


def main():
    parser = argparse.ArgumentParser("Interactive valid region visualization")
    parser.add_argument("--input_camera", type=str, required=True)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--step", type=float, default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    camera = model_selector_from_str(args.input_camera)
    camera = camera.to(device)
    w, h = [int(x.item()) for x in camera.image_shape]

    valid_mask = estimate_valid_region(camera, step=args.step)
    mask_np = valid_mask.cpu().numpy()

    # Build image
    if args.img_path is not None:
        img = cv2.imread(args.img_path)
        img = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
    else:
        img = np.full((h, w, 3), 180, dtype=np.uint8)

    # Red overlay on invalid region
    overlay = img.copy().astype(np.float32)
    invalid = ~mask_np
    overlay[invalid, 0] = overlay[invalid, 0] * 0.4 + 255 * 0.6  # red
    overlay[invalid, 1] = overlay[invalid, 1] * 0.4                # dim green
    overlay[invalid, 2] = overlay[invalid, 2] * 0.4                # dim blue
    overlay = overlay.clip(0, 255).astype(np.uint8)

    # Green boundary contour
    mask_u8 = mask_np.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    center = camera.get_center().detach()
    if center.numel() < 2:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center[0].item(), center[1].item()

    max_r = max(w, h)
    pct = 100 * mask_np.mean()

    # --- Figure ---
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.18)

    ax_img.imshow(overlay)
    ax_img.set_title(f"Valid region: {pct:.1f}%")
    ax_img.axis('off')

    # Cut line
    az0 = 0.0
    cut_line, = ax_img.plot(
        [cx, cx + max_r * math.cos(az0)],
        [cy, cy + max_r * math.sin(az0)],
        'c-', linewidth=1.5)

    # Radial profile
    radii, angles, valid = compute_radial_profile(camera, az0)
    profile_line, = ax_plot.plot(radii[valid], angles[valid], 'b-', linewidth=2)
    ax_plot.set_xlabel("Pixel radius from center")
    ax_plot.set_ylabel("Ray angle from principal axis (°)")
    ax_plot.set_title("Radial profile")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_xlim(0, max_r * 0.8)
    ax_plot.set_ylim(0, 200)

    def get_boundary_radius(azimuth):
        dx, dy = math.cos(azimuth), math.sin(azimuth)
        for r in range(1, int(max_r)):
            px, py = int(cx + r * dx), int(cy + r * dy)
            if px < 0 or px >= w or py < 0 or py >= h or not mask_np[py, px]:
                return r
        return max_r

    boundary_vline = ax_plot.axvline(
        x=get_boundary_radius(az0), color='r', linestyle='--', linewidth=1.5, label='valid boundary')
    ax_plot.legend()

    # Slider
    ax_slider = fig.add_axes((0.15, 0.05, 0.7, 0.03))
    slider = Slider(ax_slider, 'Azimuth (°)', 0, 360, valinit=0, valstep=1)

    def update(_):
        azimuth = slider.val * math.pi / 180
        cut_line.set_xdata([cx, cx + max_r * math.cos(azimuth)])
        cut_line.set_ydata([cy, cy + max_r * math.sin(azimuth)])

        radii, angles, valid = compute_radial_profile(camera, azimuth)
        if valid.any():
            profile_line.set_xdata(radii[valid])
            profile_line.set_ydata(angles[valid])

        boundary_vline.set_xdata([get_boundary_radius(azimuth)])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    main()
