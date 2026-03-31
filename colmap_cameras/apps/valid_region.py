"""
2026 Daniil Sinitsyn

Interactive visualization of camera valid region and radial distortion profile.
"""
import os
import sys
import argparse
import math

if sys.platform == 'darwin':
    import matplotlib; matplotlib.use('macosx')
elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
    import matplotlib
    try: matplotlib.use('TkAgg')
    except ImportError: matplotlib.use('Agg')

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from colmap_cameras import model_selector_from_str
from colmap_cameras import CompositeCamera
from colmap_cameras.utils.valid_region import estimate_valid_region


def radial_profile(camera, azimuth, n_samples=500):
    """Pixel radius vs ray angle along one azimuthal cut."""
    center = camera.get_center().detach()
    if center.numel() < 2:
        center = camera.image_shape.float() / 2

    max_r = max(camera.image_shape[0].item(), camera.image_shape[1].item())
    radii = torch.linspace(0, max_r, n_samples, device=camera.device)
    d = torch.tensor([math.cos(azimuth), math.sin(azimuth)], device=camera.device)
    pts2d = center + radii[:, None] * d[None, :]

    with torch.no_grad():
        rays = camera.unmap(pts2d)

    valid = ~torch.isnan(rays).any(dim=-1)
    r_norm = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta = torch.acos(r_norm[:, 2].clamp(-1, 1)) * 180 / math.pi
    return radii.cpu().numpy(), theta.cpu().numpy(), valid.cpu().numpy()


def main():
    parser = argparse.ArgumentParser("Interactive valid region visualization")
    parser.add_argument("--input_camera", type=str, required=True)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--step", type=float, default=2)
    parser.add_argument("--save-mask", type=str, default=None, help="Save valid region mask to this path")
    parser.add_argument("--visualize", action="store_true", help="Show interactive visualization")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    camera = model_selector_from_str(args.input_camera).to(device)
    comp = CompositeCamera(camera)
    comp.update_boundary()

    w, h = int(camera.image_shape[0].item()), int(camera.image_shape[1].item())
    mask_np = estimate_valid_region(camera, step=args.step).cpu().numpy()

    if args.save_mask:
        cv2.imwrite(args.save_mask, mask_np.astype(np.uint8) * 255)
        print(f"Saved mask to {args.save_mask}")

    if not args.visualize:
        return

    # Build overlay
    if args.img_path is not None:
        img = cv2.cvtColor(cv2.resize(cv2.imread(args.img_path), (w, h)), cv2.COLOR_BGR2RGB)
    else:
        img = np.full((h, w, 3), 180, dtype=np.uint8)
    overlay = img.copy().astype(np.float32)
    overlay[~mask_np, 0] = overlay[~mask_np, 0] * 0.4 + 255 * 0.6
    overlay[~mask_np, 1:] *= 0.4
    overlay = overlay.clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    center = camera.get_center().detach()
    cx = center[0].item() if center.numel() >= 2 else w / 2.0
    cy = center[1].item() if center.numel() >= 2 else h / 2.0
    max_r = max(w, h)

    # Figure
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.18)
    ax_img.imshow(overlay)
    ax_img.set_title(f"Valid region: {100 * mask_np.mean():.1f}%")
    ax_img.axis('off')

    cut_line, = ax_img.plot([cx, cx + max_r], [cy, cy], 'c-', lw=1.5)
    profile_line, = ax_plot.plot([], [], 'b-', lw=2, label='model')
    cont_line, = ax_plot.plot([], [], '--', color='orange', lw=2, label='continuation')
    ax_plot.axhline(y=180, color='gray', ls=':', alpha=0.5, label='180°')
    ax_plot.set_xlabel("Pixel radius")
    ax_plot.set_ylabel("Ray angle (°)")
    ax_plot.set_xlim(0, max_r * 0.8)
    ax_plot.set_ylim(0, 200)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend(loc='upper left')

    def update_plot(az):
        # Inner model profile
        r, a, v = radial_profile(camera, az)
        profile_line.set_data(r[v], a[v])
        # Composite camera profile (includes continuation)
        r_c, a_c, v_c = radial_profile(comp, az)
        cont_line.set_data(r_c[v_c], a_c[v_c])

    update_plot(0.0)

    ax_slider = fig.add_axes((0.15, 0.05, 0.7, 0.03))
    slider = Slider(ax_slider, 'Azimuth (°)', 0, 360, valinit=0, valstep=1)

    def on_change(_):
        az = slider.val * math.pi / 180
        cut_line.set_data([cx, cx + max_r * math.cos(az)], [cy, cy + max_r * math.sin(az)])
        update_plot(az)
        fig.canvas.draw_idle()

    slider.on_changed(on_change)
    plt.show()


if __name__ == '__main__':
    main()
