# import the necessary packages
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import argparse
import cv2

clusterImgName = []
clusterImages = []



#Select images for clustering using GUI
def openClustImages(self):
	imgNames = QFileDialog.getOpenFileNames(self, self.tr("Select image(s) to cluster"))
	for index in range(len(imgNames[0])):
		imgName = imgNames[0][index]
		self.clusterImgName.append(imgName)
		self.clusterImageWrapper(imgName)

#pull all of the images for cluster as grayscale into an index
def clusterImageWrapper(self, imgName):
    self.clusterImages.append(cv2.imread(imgName, cv2.IMREAD_GRAYSCALE))

#configure Images for kmeans and run kmeans
def kmeansCluster(self, index):
	flatImages = self.clusterImages[index]
	nImages = len(flatImages[0])
	for index in range(len(flatImages[0])):
		flatImg = flatImages [0][index]
		flatImg.reshape([-1],1)
		
	kmeansInput = (flatImg,nImages)
	kmeansInput = np.float32(kmeansInput)



