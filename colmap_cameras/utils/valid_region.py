"""
2026 Daniil Sinitsyn

Estimate the valid image region of a camera model.

Checks the round-trip Jacobian det(d map(unmap(p)) / dp) ≈ 1,
then flood-fills from the principal point.
"""
import torch


def roundtrip_jacobian_det(camera, pts2d):
    """Compute det(d map(unmap(p)) / dp).
    pts2d must have requires_grad=True.
    Returns (det, valid)."""
    rays = camera.unmap(pts2d)
    ok = ~torch.isnan(rays).any(dim=-1)
    pts2d_out, vm = camera.map(rays)
    ok = ok & vm

    det = torch.zeros(len(pts2d), device=pts2d.device)
    if ok.any():
        g0 = torch.autograd.grad(pts2d_out[:, 0].sum(), pts2d, retain_graph=True)[0]
        g1 = torch.autograd.grad(pts2d_out[:, 1].sum(), pts2d)[0]
        det = g0[:, 0] * g1[:, 1] - g0[:, 1] * g1[:, 0]
    return det, ok


def estimate_valid_region(camera, step=1.0, chunk_size=10000, rt_tolerance=0.5):
    """Estimate valid region via round-trip Jacobian, flood-filled from center.
    Returns (H, W) bool tensor."""
    device = camera.device
    w, h = [int(x.item()) for x in camera.image_shape]

    u = torch.arange(0, w, step, device=device, dtype=torch.float32)
    v = torch.arange(0, h, step, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)
    n = pts2d.shape[0]
    valid_all = torch.zeros(n, dtype=torch.bool, device=device)

    for i in range(0, n, chunk_size):
        chunk = pts2d[i:i + chunk_size].detach().requires_grad_(True)
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
