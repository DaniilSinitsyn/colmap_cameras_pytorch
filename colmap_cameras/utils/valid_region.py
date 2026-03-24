"""
2026 Daniil Sinitsyn

Estimate the valid image region where the camera projection is locally injective.
Checks the Jacobian determinant of the round-trip map(unmap(pixel))
and flood-fills from the principal point.
"""
import torch


def estimate_valid_region(camera, step=1.0, chunk_size=10000):
    """Estimate the valid region of a camera model.

    For each pixel p, computes det(d(map(unmap(p)))/dp). Where the
    distortion folds, the round-trip Jacobian determinant drops to zero
    or flips sign. The final mask is the connected component around
    the principal point where det > 0.

    Args:
        camera: a camera model instance (BaseModel subclass)
        step: pixel step for the grid (can be fractional, e.g. 0.5 for sub-pixel)
        chunk_size: process this many pixels at a time to limit memory

    Returns:
        valid_mask: (H, W) boolean tensor, True where the pixel is valid
    """
    device = camera.device
    w, h = [int(x.item()) for x in camera.image_shape]

    u = torch.arange(0, w, step, device=device, dtype=torch.float32)
    v = torch.arange(0, h, step, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1)

    n = pts2d.shape[0]
    det_all = torch.zeros(n, device=device)
    valid_all = torch.zeros(n, dtype=torch.bool, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pts2d[start:end].detach().requires_grad_(True)

        pts3d = camera.unmap(chunk)
        valid_unmap = ~torch.isnan(pts3d).any(dim=-1)
        pts2d_out, valid_map = camera.map(pts3d)

        ok = valid_unmap & valid_map

        # Round-trip error: catches root-switching in Newton solvers
        err = (pts2d_out - chunk).detach().norm(dim=-1)
        ok = ok & (err < 1.0)

        # Jacobian sign: catches smooth folds
        det = torch.zeros(end - start, device=device)
        if ok.any():
            g0 = torch.autograd.grad(pts2d_out[:, 0].sum(), chunk, retain_graph=True)[0]
            g1 = torch.autograd.grad(pts2d_out[:, 1].sum(), chunk)[0]
            det = g0[:, 0] * g1[:, 1] - g0[:, 1] * g1[:, 0]

        det_all[start:end] = det.detach()
        valid_all[start:end] = ok

    locally_valid = valid_all & (det_all > 0)

    ws, hs = len(u), len(v)
    local_grid = locally_valid.reshape(ws, hs).T

    center = camera.get_center().detach()
    if center.numel() >= 2:
        cu = int(round(center[0].item() / step))
        cv = int(round(center[1].item() / step))
    else:
        cu, cv = ws // 2, hs // 2
    cu = max(0, min(cu, ws - 1))
    cv = max(0, min(cv, hs - 1))

    sr, sc = _find_valid_start(local_grid, cv, cu)
    valid_grid = _flood_fill(local_grid, sr, sc)

    if valid_grid.shape != (h, w):
        valid_grid = valid_grid.unsqueeze(0).unsqueeze(0).float()
        valid_mask = torch.nn.functional.interpolate(
            valid_grid, size=(h, w), mode='nearest').squeeze().bool()
    else:
        valid_mask = valid_grid

    return valid_mask


def _find_valid_start(grid, row, col):
    h, w = grid.shape
    if grid[row, col]:
        return row, col
    for radius in range(1, max(h, w)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < h and 0 <= c < w and grid[r, c]:
                    return r, c
    return row, col


def _flood_fill(grid, start_row, start_col):
    h, w = grid.shape
    result = torch.zeros_like(grid)
    if not grid[start_row, start_col]:
        return result

    grid_np = grid.cpu().numpy()
    visited = result.cpu().numpy()

    queue = [(start_row, start_col)]
    visited[start_row, start_col] = True

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid_np[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return torch.from_numpy(visited).to(grid.device)
