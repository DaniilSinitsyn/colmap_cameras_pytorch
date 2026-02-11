"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import unittest
from .fov import TestFov
from .full_opencv  import TestFullOpenCV
from .opencv import TestOpenCV
from .opencv_fisheye import TestOpenCVFisheye
from .pinhole import TestPinhole
from .simple_pinhole import TestSimplePinhole
from .radial import TestRadial
from .simple_radial import TestSimpleRadial
from .simple_radial_fisheye import TestSimpleRadialFisheye
from .radial_fisheye import TestRadialFisheye
from .thin_prism_fisheye import TestThinPrismFisheye
from .division_model import TestDivisionModel
from .polynomial_division_model import TestPolynomialDivisionModel
from .unified_camera_model import TestUnifiedCameraModel
from .meis_camera_model import TestMeisCameraModel
from .equirectangular import TestEquirectangular

if __name__ == '__main__':
    unittest.main()
