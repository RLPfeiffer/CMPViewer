from PyQt5.QtGui import QImage
from PyQt5.QtGui import QColor
import code
from qimage2ndarray import *
import time
import cv2
import numpy as np

def create_composite_image(rawImages, r_image, g_image, b_image):
    # Get numpy representation of images for optimized color manipulation
    shape = rawImages[0].shape[:2]

    r = r_image
    g = g_image
    b = b_image

    if r is None:
        r = np.zeros(shape=shape, dtype=np.uint8)
    if g is None:
        g = np.zeros(shape=shape, dtype=np.uint8)
    if b is None:
        b = np.zeros(shape=shape, dtype=np.uint8)

    # Convert our composite back to QImage
    return array2qimage(np.dstack((r, g, b)))

