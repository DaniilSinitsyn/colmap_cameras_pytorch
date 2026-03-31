"""
2026 Daniil Sinitsyn

Estimate the valid image region of a camera model.

Checks the round-trip Jacobian det(d map(unmap(p)) / dp) ≈ 1
via finite differences, then flood-fills from the principal point.
"""
import torch


def roundtrip_jacobian_det(camera, pts2d, eps=0.5):
    """Compute det(d map(unmap(p)) / dp) via finite differences.
    No autograd needed. Returns (det, valid)."""
    du = torch.tensor([eps, 0], device=pts2d.device)
    dv = torch.tensor([0, eps], device=pts2d.device)

    rays = camera.unmap(pts2d)
    if isinstance(rays, tuple):
        rays = rays[0]
    g, v0 = camera.map(rays)

    rays_du = camera.unmap(pts2d + du)
    if isinstance(rays_du, tuple):
        rays_du = rays_du[0]
    g_du, v1 = camera.map(rays_du)

    rays_dv = camera.unmap(pts2d + dv)
    if isinstance(rays_dv, tuple):
        rays_dv = rays_dv[0]
    g_dv, v2 = camera.map(rays_dv)

    ok = (~torch.isnan(rays).any(dim=-1) & v0
          & ~torch.isnan(rays_du).any(dim=-1) & v1
          & ~torch.isnan(rays_dv).any(dim=-1) & v2)

    dg_du = (g_du - g) / eps
    dg_dv = (g_dv - g) / eps
    det = dg_du[:, 0] * dg_dv[:, 1] - dg_du[:, 1] * dg_dv[:, 0]
    return det, ok


def estimate_valid_region(camera, step=1.0, chunk_size=50000, rt_tolerance=0.5):
    """Estimate valid region via round-trip Jacobian, flood-filled from center.
    Uses finite differences — no autograd, works with torch.no_grad().
    Returns (H, W) bool tensor."""
    device = camera.device
    w, h = [int(x.item()) for x in camera.image_shape]

    u = torch.arange(0, w, step, device=device, dtype=torch.float32)
    v = torch.arange(0, h, step, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)
    n = pts2d.shape[0]
    valid_all = torch.zeros(n, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(0, n, chunk_size):
            chunk = pts2d[i:i + chunk_size]
            det, ok = roundtrip_jacobian_det(camera, chunk)
            valid_all[i:i + len(chunk)] = ok & ((det - 1.0).abs() < rt_tolerance)

    ws, hs = len(u), len(v)
    grid = valid_all.reshape(ws, hs).T

    center = camera.get_center().detach()
    cu = int(round(center[0].item() / step)) if center.numel() >= 2 else ws // 2
    cv = int(round(center[1].item() / step)) if center.numel() >= 2 else hs // 2
    cu, cv = max(0, min(cu, ws - 1)), max(0, min(cv, hs - 1))

    sr, sc = _find_start(grid, cv, cu)
    out = _flood_fill(grid, sr, sc)

    if out.shape != (h, w):
        out = torch.nn.functional.interpolate(
            out[None, None].float(), size=(h, w), mode='nearest').squeeze().bool()
    return out


def _find_start(grid, row, col):
    h, w = grid.shape
    if grid[row, col]:
        return row, col
    for r in range(1, max(h, w)):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc]:
                    return nr, nc
    return row, col


def _flood_fill(grid, sr, sc):
    h, w = grid.shape
    out = torch.zeros_like(grid)
    if not grid[sr, sc]:
        return out
    g = grid.cpu().numpy()
    v = out.cpu().numpy()
    q = [(sr, sc)]
    v[sr, sc] = True
    while q:
        r, c = q.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not v[nr, nc] and g[nr, nc]:
                v[nr, nc] = True
                q.append((nr, nc))
    return torch.from_numpy(v).to(grid.device)
