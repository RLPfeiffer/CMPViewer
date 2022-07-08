import numpy as np
import nornir_imageregistration

class ImageSet:

    @property
    def selectedImages(self):
        return self.images[self.selected,:,:]
		
    @property
    def numImages(self):
        return self.images.shape[0]

    def __init__(self, images: np.ndarray):
        self.images = images
        self.selected = np.bool8((images.shape[0],))
    

#configure Images for kmeans and run kmeans
def kmeansCluster(input: ImageSet, index):
    """
    :param ImageSet input: The ImageSet to apply clustering to
    :param index: A mask or index array indicating which images to include in clustering
    """
    clusterImages = input[index,:,:]
    nImages = clusterImages.shape[0]
    for index in range(len(flatImages[0])):
        flatImg = flatImages [0][index]
        flatImg.reshape([-1],1)
		
    kmeansInput = (flatImg,nImages)
    kmeansInput = np.float32(kmeansInput)
