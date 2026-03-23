"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_pt2d(model, test_self):
    model = model.to(DEVICE)
    mw, mh = (model.image_shape * 0.22).int()
    u = torch.arange(mw, model.image_shape[0]-mw, 10, device=DEVICE)
    v = torch.arange(mh, model.image_shape[1]-mh, 10, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1).float()

    pts3d = model.unmap(pts2d)
    pts2d_new, valid = model.map(pts3d)

    diff = (pts2d_new - pts2d).abs()
    test_self.assertTrue(valid.all())
    test_self.assertLess(diff[:, 0].max().item(), 1e-2)
    test_self.assertLess(diff[:, 1].max().item(), 1e-2)

def test_pt3d(model, test_self):
    model = model.to(DEVICE)
    u = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
    v = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts3d = torch.stack([uu.ravel(), vv.ravel(), torch.ones_like(uu.ravel())], dim=-1).float()
    pts3d /= torch.norm(pts3d, dim=-1, keepdim=True)

    pts2d, valid = model.map(pts3d)
    pts3d_new = model.unmap(pts2d)
    pts3d_new /= torch.norm(pts3d_new, dim=-1, keepdim=True)

    diff = (pts3d_new - pts3d).abs()
    test_self.assertTrue(valid.all())
    test_self.assertLess(diff[:, 0].max().item(), 1e-4)
    test_self.assertLess(diff[:, 1].max().item(), 1e-4)
    test_self.assertLess(diff[:, 2].max().item(), 1e-4)

def test_model_fit(model1, model2, iters, test_self):
    """Fit model2 to model1 using torch.optim (2D reprojection error)."""
    model1 = model1.to(DEVICE)
    model2 = model2.to(DEVICE)

    u = torch.arange(0, model1.image_shape[0], 5, device=DEVICE)
    v = torch.arange(0, model1.image_shape[1], 5, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    points2d = torch.stack([uu.ravel(), vv.ravel()], dim=-1).float()

    with torch.no_grad():
        pts3d = model1.unmap(points2d)

    model2.requires_grad = True
    n_iters = iters * 30
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.05)

    for _ in range(n_iters):
        optimizer.zero_grad()
        pts2d_hat, valid = model2.map(pts3d)
        if valid.sum() == 0: continue
        loss = ((points2d[valid] - pts2d_hat[valid]) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
        v = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        points3d = torch.stack([uu.ravel(), vv.ravel(), torch.ones_like(uu.ravel())], dim=-1).float()

        pts2d_1, valid_1 = model1.map(points3d)
        pts2d_2, valid_2 = model2.map(points3d)
        valid = valid_1 & valid_2

        diff = (pts2d_1[valid] - pts2d_2[valid]).abs()
        if diff.numel() > 0:
            test_self.assertLess(diff[:, 0].max().item(), 1)
            test_self.assertLess(diff[:, 1].max().item(), 1)

def test_model_3dpts_fil(model1, model2, test_self):
    """Fit model2 to model1 using torch.optim (3D ray error)."""
    model1 = model1.to(DEVICE)
    model2 = model2.to(DEVICE)

    u = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
    v = torch.arange(-0.5, 0.5, 0.1, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    points3d = torch.stack([uu.ravel(), vv.ravel(), torch.ones_like(uu.ravel())], dim=-1).float()
    points3d /= torch.norm(points3d, dim=-1, keepdim=True)

    with torch.no_grad():
        pts2d, valid = model1.map(points3d)
        pts2d = pts2d[valid]
        target_3d = points3d[valid]

    with torch.no_grad():
        pts3d_hat = model2.unmap(pts2d)
        pts3d_hat = pts3d_hat / torch.norm(pts3d_hat, dim=-1, keepdim=True)
        first_loss = ((pts3d_hat - target_3d) ** 2).mean()

    model2.requires_grad = True
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for _ in range(100):
        optimizer.zero_grad()
        pts3d_hat = model2.unmap(pts2d)
        pts3d_hat = pts3d_hat / torch.norm(pts3d_hat, dim=-1, keepdim=True)
        loss = ((pts3d_hat - target_3d) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        pts3d_hat = model2.unmap(pts2d)
        pts3d_hat = pts3d_hat / torch.norm(pts3d_hat, dim=-1, keepdim=True)
        final_loss = ((pts3d_hat - target_3d) ** 2).mean()

    test_self.assertLess(final_loss, first_loss)
