import typing
from PIL import Image
import numpy
from numpy.typing import NDArray

def numpy_labels_to_pillow_image(input: NDArray[int]) -> Image:
    # Create a new image with the same size as the original image
    output_img = Image.new('P', (input.shape[1], input.shape[0]))
    output_img.putdata(numpy.array(input.flat))
    return output_img