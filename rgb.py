from PyQt5.QtGui import QImage
import code

if __name__ == "__main__":
    image = QImage("test_images/7564_12191_RLP_40x_05_TT.tif.tif")
    code.interact(local=locals())

# def get_image(r_g_or_b):
#
#
# def create_composite_image():
#   r_image = get_image(r)
#   g_image = get_image(g)
#   b_image = get_image(b)
#   
#   our_composite_image = r_image.copy()
#
#   We assume that every image has the same dimensions! So it doesn't
#   matter which one we loop:
#   for x in range(r_image.width()):
#       for y in range(r_image.height()):
#           # We need to calculate this pixel's composite color
#           r = r_image.pixelColor(x, y).value()
#           g = g_image.pixelColor(x, y).value()
#           b = b_image.pixelColor(x, y).value()
#           new_color = QColor(r, g, b)
#           our_composite_image.setPixelColor(x, y, new_color)
#
#   return our_composite_image