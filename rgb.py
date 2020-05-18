from PyQt5.QtGui import QImage
from PyQt5.QtGui import QColor
import code
from qimage2ndarray import *
import time

def create_composite_image(r_image, g_image, b_image):    
    # Get numpy representation of images for optimized color manipulation
    r_optimized = rgb_view(r_image)
    g_optimized = rgb_view(g_image)
    b_optimized = rgb_view(b_image)

    # Make an ndarray for the composite
    composite_optimized = r_optimized.copy()

    # Assign the composite pixels
    for x in range(r_image.width()):
        for y in range(r_image.height()):
            # We need to calculate this pixel's composite color
            r = r_optimized[y, x, 0]
            g = g_optimized[y, x, 0]
            b = b_optimized[y, x, 0]

            composite_optimized[y, x, 0] = r
            composite_optimized[y, x, 1] = g
            composite_optimized[y, x, 2] = b
    
    # Convert our composite back to QImage
    return array2qimage(composite_optimized)

# def chooseRedImage(rImage):
#     r_image = rImage
#     create_composite_image(r_image)
#     print('r_image' in locals())
#     #code.interact(local=locals())

# def chooseGreenImage(gImage):
#     g_image = gImage
#     create_composite_image(g_image)
#     print('g_image' in locals())
#     #code.interact(local=locals())
#
# def chooseBlueImage(bImage):
#     b_image = bImage
#     create_composite_image(b_image)
#     print('b_image' in locals())
#     #code.interact(local=locals())


if __name__ == "__main__":
    # Load images and convert them to numpy-compatible pixel format.
    r_image = QImage("test_images/7564_12191_RLP_40x_05_TT.tif.tif").convertToFormat(QImage.Format_RGB32)
    g_image = QImage("test_images/7564_12191_RLP_40x_04_Q.tif.tif").convertToFormat(QImage.Format_RGB32)
    b_image = QImage("test_images/7564_12191_RLP_40x_06_E.tif.tif").convertToFormat(QImage.Format_RGB32)
    #chooseRedImage(rImage)
    print("Compositing started")
    time_start = time.perf_counter_ns()
    our_composite = create_composite_image(r_image, g_image, b_image)
    time_end = time.perf_counter_ns()
    print('It took {} nanoseconds to composite'.format(time_end - time_start))

    #chooseGreenImage(gImage)
    #chooseBlueImage(bImage)
    code.interact(local=locals())
