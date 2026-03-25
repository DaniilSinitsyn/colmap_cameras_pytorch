"""
2026 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_grid_2d(w, h, step, device):
    u = torch.arange(0, w, step, device=device)
    v = torch.arange(0, h, step, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    return torch.stack([uu.ravel(), vv.ravel()], dim=-1).float()


def _make_grid_3d(device):
    u = torch.arange(-0.5, 0.5, 0.1, device=device)
    v = torch.arange(-0.5, 0.5, 0.1, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    pts = torch.stack([uu.ravel(), vv.ravel(), torch.ones_like(uu.ravel())], dim=-1).float()
    return pts / pts.norm(dim=-1, keepdim=True)


def test_pt2d(model, test_self):
    model = model.to(DEVICE)
    mw, mh = (model.image_shape * 0.22).int()
    u = torch.arange(mw, model.image_shape[0] - mw, 10, device=DEVICE)
    v = torch.arange(mh, model.image_shape[1] - mh, 10, device=DEVICE)
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
    pts3d = _make_grid_3d(DEVICE)

    pts2d, valid = model.map(pts3d)
    pts3d_new = model.unmap(pts2d)
    pts3d_new = pts3d_new / pts3d_new.norm(dim=-1, keepdim=True)
    diff = (pts3d_new - pts3d).abs()

    test_self.assertTrue(valid.all())
    for i in range(3):
        test_self.assertLess(diff[:, i].max().item(), 1e-4)


def test_model_fit(model1, model2, iters, test_self):
    model1, model2 = model1.to(DEVICE), model2.to(DEVICE)
    w, h = int(model1.image_shape[0].item()), int(model1.image_shape[1].item())
    pts2d = _make_grid_2d(w, h, 5, DEVICE)

    with torch.no_grad():
        pts3d = model1.unmap(pts2d)

    model2.requires_grad = True
    opt = torch.optim.Adam(model2.parameters(), lr=0.05)
    for _ in range(iters * 30):
        opt.zero_grad()
        pred, valid = model2.map(pts3d)
        if valid.sum() == 0:
            continue
        ((pts2d[valid] - pred[valid]) ** 2).mean().backward()
        opt.step()

    with torch.no_grad():
        pts3d_test = _make_grid_3d(DEVICE) * 3
        p1, v1 = model1.map(pts3d_test)
        p2, v2 = model2.map(pts3d_test)
        valid = v1 & v2
        if valid.any():
            diff = (p1[valid] - p2[valid]).abs()
            test_self.assertLess(diff[:, 0].max().item(), 1)
            test_self.assertLess(diff[:, 1].max().item(), 1)


def test_model_3dpts_fil(model1, model2, test_self):
    model1, model2 = model1.to(DEVICE), model2.to(DEVICE)
    pts3d = _make_grid_3d(DEVICE)

    with torch.no_grad():
        pts2d, valid = model1.map(pts3d)
        pts2d, target = pts2d[valid], pts3d[valid]

    with torch.no_grad():
        pred = model2.unmap(pts2d)
        first_loss = ((pred / pred.norm(dim=-1, keepdim=True) - target) ** 2).mean()

    model2.requires_grad = True
    opt = torch.optim.Adam(model2.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    for _ in range(100):
        opt.zero_grad()
        pred = model2.unmap(pts2d)
        loss = ((pred / pred.norm(dim=-1, keepdim=True) - target) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        pred = model2.unmap(pts2d)
        final_loss = ((pred / pred.norm(dim=-1, keepdim=True) - target) ** 2).mean()

    test_self.assertLess(final_loss, first_loss)
