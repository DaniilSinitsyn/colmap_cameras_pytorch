"""
2026 Daniil Sinitsyn

Base class for camera wrappers that delegate to an inner model.
"""
import torch


class CameraAdapter(torch.nn.Module):
    """Base for cameras that wrap an inner model. Delegates all common
    methods to self.inner. Subclasses override map/unmap as needed."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def map(self, points3d):       return self.inner.map(points3d)
    def unmap(self, points2d):     return self.inner.unmap(points2d)
    def get_center(self):          return self.inner.get_center()
    def get_fov(self):             return self.inner.get_fov()
    def to_colmap(self):           return self.inner.to_colmap()
    def fix(self, *p):             return self.inner.fix(*p)
    def unfix(self, *p):           return self.inner.unfix(*p)

    @property
    def fixed(self):               return self.inner.fixed
    @property
    def image_shape(self):         return self.inner.image_shape
    @property
    def device(self):              return self.inner.device

    @property
    def requires_grad(self):       return self.inner.requires_grad
    @requires_grad.setter
    def requires_grad(self, v):    self.inner.requires_grad = v

    @property
    def model_name(self):          return self.inner.model_name

    def __repr__(self):            return repr(self.inner)
