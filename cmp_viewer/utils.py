import typing
from PIL import Image
import numpy
from numpy.typing import NDArray

"""
This module provides utility functions for image processing and conversion
between different image formats used in the CMP Viewer application.
"""

def numpy_labels_to_pillow_image(input: NDArray[int]) -> Image:
    """
    Convert a NumPy array of integer labels to a PIL Image in palette mode.

    This function is particularly useful for visualizing clustered images where
    each pixel value represents a cluster label. The resulting image uses the
    'P' (palette) mode, which is suitable for images with a limited number of colors.

    Args:
        input (NDArray[int]): A 2D NumPy array containing integer labels.
                             Shape should be (height, width).

    Returns:
        Image: A PIL Image object in palette mode ('P') with the same dimensions
              as the input array. Each pixel value corresponds to the label in
              the input array.

    Note:
        The palette of the output image is not set by this function. To visualize
        the labels with distinct colors, you may need to set a custom palette.
    """
    # Create a new image with the same size as the original image
    output_img = Image.new('P', (input.shape[1], input.shape[0]))
    output_img.putdata(numpy.array(input.flat))
    return output_img
