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
    def __init__(self, fileNameList, checked=False):
        """
        :param fileNameList: A list of strings that contain the names of each entry in the widget
        :param checked: An array of true/false values that set the starting state of the selected checkbox
        """
        super().__init__()
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        self.items = []
        #self.list.addItems(fileNameList)
        for (i, fileName) in enumerate(fileNameList):
            item = QtWidgets.QListWidgetItem(fileName)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.data = i
            check = Qt.Checked if checked[i] else Qt.Unchecked
            item.setCheckState(check)
            self.clusterList.addItem(item)
            self.items.append(item)
        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        self.selectedMask = np.zeros((len(fileNameList)), bool)
        
        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Select")
        self.button1.clicked.connect(self.clusterOptions)
        
        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def clusterOptions(self, whoknows):
        for item in self.items:
            if item.checkState() == Qt.Checked:
                self.selectedMask[item.data] = True
                #self.clusterImgName.append(item)
                #self.clusterImages.append(self.rawImages[item.data])

#pull all of the images for cluster as grayscale into an index
def clusterImageWrapper(self, imgName):
    self.clusterImages.append(cv2.imread(imgName, cv2.IMREAD_GRAYSCALE))



