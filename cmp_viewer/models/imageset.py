import numpy as np
from numpy.typing import NDArray


class ImageSet:

    @property
    def num_images(self) -> int:
        return self._raw_images.shape[0]

    @property
    def images(self) -> NDArray[float]:
        return self._raw_images

    @property
    def image_shape(self) -> NDArray[int]:
        return self._raw_images.shape[1:]

    def __init__(self):
        self._raw_images = None  #type: NDArray
        pass

    def add_image(self, img: NDArray):
        if not np.issubdtype(img.dtype, np.floating):
            raise NotImplementedError("Expecting a floating point NDArray here")

        reshaped_img = img.reshape((1, img.shape[0], img.shape[1]))

        if self._raw_images is None:
            self._raw_images = reshaped_img
        else:
            self._raw_images = np.vstack((self._raw_images, reshaped_img))