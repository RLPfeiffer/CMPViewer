import numpy as np
from numpy.typing import NDArray

"""
This module contains the ImageSet class for managing collections of multidimensional images.
"""

class ImageSet:
    """
    A class for managing a collection of multidimensional images stored as NumPy arrays.

    The ImageSet class provides functionality to store multiple images and access them
    as a single NumPy array. It ensures all images have the same dimensions and are
    stored as floating-point arrays.
    """

    @property
    def num_images(self) -> int:
        """
        Get the number of images in the set.

        Returns:
            int: The number of images currently stored in the set.
        """
        return self._raw_images.shape[0]

    @property
    def images(self) -> NDArray[float]:
        """
        Get the raw image data.

        Returns:
            NDArray[float]: A NumPy array containing all images in the set.
                           Shape is (num_images, height, width).
        """
        return self._raw_images

    @property
    def image_shape(self) -> NDArray[int]:
        """
        Get the shape of the images (excluding the batch dimension).

        Returns:
            NDArray[int]: The shape of each image (height, width).
        """
        return self._raw_images.shape[1:]

    def __init__(self):
        """
        Initialize an empty ImageSet.

        The _raw_images attribute will be None until images are added.
        """
        self._raw_images = None  #type: NDArray
        pass

    def add_image(self, img: NDArray):
        """
        Add a new image to the set.

        Args:
            img (NDArray): The image to add. Must be a floating-point NumPy array.
                          Will be reshaped to (1, height, width) if necessary.

        Raises:
            NotImplementedError: If the image is not a floating-point array.
        """
        if not np.issubdtype(img.dtype, np.floating):
            raise NotImplementedError("Expecting a floating point NDArray here")

        # Reshape the image to ensure it has the correct dimensions (1, height, width)
        reshaped_img = img.reshape((1, img.shape[0], img.shape[1]))

        # Initialize _raw_images if this is the first image, otherwise stack the new image
        if self._raw_images is None:
            self._raw_images = reshaped_img
        else:
            self._raw_images = np.vstack((self._raw_images, reshaped_img))
