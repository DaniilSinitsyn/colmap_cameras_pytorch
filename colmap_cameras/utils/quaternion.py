"""
2026 Daniil Sinitsyn

Quaternion utilities. Convention: (qw, qx, qy, qz), Hamilton product. Purely CLAUDE generated
"""
import torch


def quat_to_rotmat(q):
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

    Args:
        q: (..., 4) tensor.

    Returns:
        (..., 3, 3) rotation matrix.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy - qz * qw),
        2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw),
        1 - 2 * (qx * qx + qz * qz),
        2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw),
        2 * (qy * qz + qx * qw),
        1 - 2 * (qx * qx + qy * qy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz).

    Uses Shepperd's method for numerical stability.

    Args:
        R: (..., 3, 3) rotation matrix.

    Returns:
        (..., 4) quaternion with positive qw.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    diag = torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2], trace], dim=-1)
    idx = diag.argmax(dim=-1)

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 0: R[0,0] is largest
    m = idx == 0
    if m.any():
        r = R[m]
        s = 2 * torch.sqrt(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2])
        q[m, 1] = s / 4
        q[m, 0] = (r[:, 2, 1] - r[:, 1, 2]) / s
        q[m, 2] = (r[:, 0, 1] + r[:, 1, 0]) / s
        q[m, 3] = (r[:, 0, 2] + r[:, 2, 0]) / s

    # Case 1: R[1,1] is largest
    m = idx == 1
    if m.any():
        r = R[m]
        s = 2 * torch.sqrt(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2])
        q[m, 2] = s / 4
        q[m, 0] = (r[:, 0, 2] - r[:, 2, 0]) / s
        q[m, 1] = (r[:, 0, 1] + r[:, 1, 0]) / s
        q[m, 3] = (r[:, 1, 2] + r[:, 2, 1]) / s

    # Case 2: R[2,2] is largest
    m = idx == 2
    if m.any():
        r = R[m]
        s = 2 * torch.sqrt(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1])
        q[m, 3] = s / 4
        q[m, 0] = (r[:, 1, 0] - r[:, 0, 1]) / s
        q[m, 1] = (r[:, 0, 2] + r[:, 2, 0]) / s
        q[m, 2] = (r[:, 1, 2] + r[:, 2, 1]) / s

    # Case 3: trace is largest
    m = idx == 3
    if m.any():
        r = R[m]
        s = 2 * torch.sqrt(1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2])
        q[m, 0] = s / 4
        q[m, 1] = (r[:, 2, 1] - r[:, 1, 2]) / s
        q[m, 2] = (r[:, 0, 2] - r[:, 2, 0]) / s
        q[m, 3] = (r[:, 1, 0] - r[:, 0, 1]) / s

    # Ensure positive qw
    q = q * q[:, 0:1].sign()

    return q.reshape(*batch_shape, 4)
