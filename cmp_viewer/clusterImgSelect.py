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

        # Add selection buttons
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        layout.addWidget(select_all_button)

        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(self.select_none)
        layout.addWidget(select_none_button)

        invert_button = QPushButton("Invert Selection")
        invert_button.clicked.connect(self.invert_selection)
        layout.addWidget(invert_button)

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)
        self.selected_mask = np.zeros((image_set.num_images), dtype=bool)

    def select_all(self):
        """Select all images for clustering and close the dialog"""
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            item.setCheckState(Qt.Checked)
        self.selected_mask[:] = True
        self.accept()

    def select_none(self):
        """Deselect all images for clustering"""
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            item.setCheckState(Qt.Unchecked)
        self.selected_mask[:] = False

    def invert_selection(self):
        """Invert the selection of all images for clustering"""
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            current_state = item.checkState() == Qt.Checked
            item.setCheckState(Qt.Checked if not current_state else Qt.Unchecked)
        self.selected_mask[:] = np.logical_not(self.selected_mask)

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