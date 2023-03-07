# import the necessary packages

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QListWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import numpy as np
import ImageViewer
from Cluster import Cluster


class clusterSelect(QtWidgets.QWidget):
    clusterImgName = []
    clusterImages = []
    """
    new popup window to select the images to be used for clustering
    """

    def __init__(self, fileNameList, rawImages, checked=False):
        super().__init__()
        self.cluster = None
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        # self.list.addItems(fileNameList)
        self.rawImages = rawImages
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
        self.button1.clicked.connect(lambda: self.clusterOptions(self.clusterList.currentRow()))

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def clusterOptions(self, index):
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            if item.checkState() == Qt.Checked:
                self.clusterImgName.append(item.text())
                self.clusterImages.append(self.rawImages[i])
        print(self.clusterImgName)
        print(self.clusterImages)
        self.cluster = Cluster(self.clusterImgName, self.clusterImages)
        self.close()
        self.cluster.show()


# configure Images for kmeans and run kmeans
def kmeansCluster(self, index):
    flatImages = self.clusterImages[index]
    nImages = len(flatImages[0])
    for index in range(len(flatImages[0])):
        flatImg = flatImages[0][index]
        flatImg.reshape([-1], 1)

    kmeansInput = (flatImg, nImages)
    kmeansInput = np.float32(kmeansInput)
