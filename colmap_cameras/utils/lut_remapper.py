"""
2026 Daniil Sinitsyn

Fast image remapping from pre-computed LUT files.
"""
import cv2
import numpy as np


class LutRemapper:
    """Remap images using pre-computed LUTs.

    Usage::
        remapper = LutRemapper('lut.npz')
        undistorted = remapper.remap('image.png')
    """

    def __init__(self, path):
        data = np.load(path)
        self.xlut = data['xlut']
        self.ylut = data['ylut']
        self.input_size = (int(data['input_size'][0]), int(data['input_size'][1]))

    def remap(self, img, interpolation=cv2.INTER_LINEAR):
        if isinstance(img, str):
            img = cv2.imread(img)
        h, w = img.shape[:2]
        if (w, h) != self.input_size:
            raise ValueError(f"Image size ({w}, {h}) != LUT input size {self.input_size}")
        return cv2.remap(img, self.xlut, self.ylut, interpolation)
