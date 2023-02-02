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

# if __name__ == "__main__":
#     # Load images and convert them to numpy-compatible pixel format.
#     r_image = QImage("test_images/7564_12191_RLP_40x_05_TT.tif.tif").convertToFormat(QImage.Format_RGB32)
#     g_image = QImage("test_images/7564_12191_RLP_40x_04_Q.tif.tif").convertToFormat(QImage.Format_RGB32)
#     b_image = QImage("test_images/7564_12191_RLP_40x_06_E.tif.tif").convertToFormat(QImage.Format_RGB32)
#     #chooseRedImage(rImage)
#     print("Compositing started")
#     # time_start = time.perf_counter_ns()
#     our_composite = create_composite_image(r_image, g_image, b_image)
#     time_end = time.perf_counter_ns()
#     print('It took {} nanoseconds to composite'.format(time_end - time_start))
#
#     #chooseGreenImage(gImage)
#     #chooseBlueImage(bImage)
#     code.interact(local=locals())
