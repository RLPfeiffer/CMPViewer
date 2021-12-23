# import the necessary packages
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMenuBar
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QListView
from PyQt5.QtWidgets import QListWidget
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QListWidgetItem
import numpy as np
import argparse
import cv2
	 
class clusterSelect(QtWidgets.QWidget):
    clusterImgName = []
    clusterImages = []
    """
    new popup window to select the images to be used for clustering
    """
    def __init__(self,fileNameList, checked=False):
        super().__init__()
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        #self.list.addItems(fileNameList)
        for items in fileNameList:
            item = QtWidgets.QListWidgetItem(items)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            check = Qt.Checked if checked else Qt.Unchecked
            item.setCheckState(check)
            self.clusterList.addItem(item)
        self.setWindowFlags(Qt.Dialog | Qt.Tool)
        
        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Select")
        self.button1.clicked.connect(self.clusterOptions)
        
        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def clusterOptions(self):
        for item in range(self.clusterList()):
            if item.checkState() == Qt.Checked:
                self.clusterImgName.append(item)

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



