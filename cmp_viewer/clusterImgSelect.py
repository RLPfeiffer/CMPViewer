# import the necessary packages

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QListWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import numpy as np
from numpy.typing import NDArray
from cmp_viewer.Cluster import Cluster
from cmp_viewer.models import ImageSet


class ImageSelectDlg(QtWidgets.QDialog):
    clusterImgName = []
    clusterImages = []

    """
    new popup window to select the images to be used for clustering
    """

    def __init__(self, fileNameList, image_set: ImageSet, checked=False, **kwargs):
        super().__init__(**kwargs)
        self.cluster = None
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        # self.list.addItems(fileNameList)
        self._image_set = image_set
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
        self.button1.clicked.connect(lambda: self.return_results(self.clusterList.currentRow()))

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)
        self.selected_mask = np.zeros((image_set.num_images), dtype=bool)

    def return_results(self, index):
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            self.selected_mask[i] = item.checkState() == Qt.Checked

        self.accept()


    def clusterOptions(self, index):
        mask = np.zeros((self.clusterList.count()))

        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            mask[i] = item.checkState() == Qt.Checked
            #if item.checkState() == Qt.Checked:
#                self.clusterImgName.append(item.text())
#                self.clusterImages.append(self.rawImages[i])
#        print(self.clusterImgName)
#        print(self.clusterImages)
#        self.cluster = Cluster(self.clusterImgName, self.clusterImages)
        self.cluster = Cluster(None, self._image_set, mask)
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
