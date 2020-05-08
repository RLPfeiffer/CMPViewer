from PyQt5.QtGui import QImage
import code

# r_image = None
# g_image = None
# b_image = None

def create_composite_image(r_image, g_image, b_image):
    rgb_image = r_image.copy()
    print('rgb_image' in locals())
    for x in range(r_image.width()):
        for y in range(r_image.height()):
              # We need to calculate this pixel's composite color
              r = r_image.pixelColor(x, y).value()
              g = g_image.pixelColor(x, y).value()
              b = b_image.pixelColor(x, y).value()
              new_color = QColor(r, g, b)
              rgb_image.setPixelColor(x, y, new_color)

        return rgb_image

    code.interact(local=locals())

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
    r_image = QImage("test_images/7564_12191_RLP_40x_05_TT.tif.tif")
    g_image = QImage("test_images/7564_12191_RLP_40x_04_Q.tif.tif")
    b_image = QImage("test_images/7564_12191_RLP_40x_06_E.tif.tif")
    #chooseRedImage(rImage)
    create_composite_image(r_image, g_image, b_image)
    #chooseGreenImage(gImage)
    #chooseBlueImage(bImage)
    #code.interact(local=locals())

# def get_image(r):


#def create_composite_image():

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
