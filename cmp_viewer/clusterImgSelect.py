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

"""
This module provides a dialog for selecting images to be used in clustering operations.
It allows users to select multiple images from a list and launch the clustering process
on the selected images.
"""

class ImageSelectDlg(QtWidgets.QDialog):
    """
    A dialog window for selecting images to be used for clustering.

    This dialog presents a list of images with checkboxes, allowing the user to
    select which images should be included in the clustering process. It provides
    buttons for selecting all images, selecting none, or inverting the current selection.

    Attributes:
        clusterImgName (list): List of names of selected images.
        clusterImages (list): List of selected image data.
        clusterList (QListWidget): Widget displaying the list of images with checkboxes.
        selected_mask (NDArray[bool]): Boolean mask indicating which images are selected.
    """
    clusterImgName = []
    clusterImages = []

    def __init__(self, fileNameList, image_set: ImageSet, checked=False, **kwargs):
        """
        Initialize the image selection dialog.

        Args:
            fileNameList (list): List of image filenames to display.
            image_set (ImageSet): Set of images that can be selected for clustering.
            checked (bool, optional): Whether all images should be checked by default. Defaults to False.
            **kwargs: Additional arguments to pass to the QDialog constructor.
        """
        super().__init__(**kwargs)
        self.cluster = None
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        # self.list.addItems(fileNameList)
        self._image_set = image_set

        # Create list items with checkboxes for each image
        for items in fileNameList:
            item = QtWidgets.QListWidgetItem(items)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            check = Qt.Checked if checked else Qt.Unchecked
            item.setCheckState(check)
            self.clusterList.addItem(item)

        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        # Create Select button
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

        # Add widgets to layout
        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

        # Initialize selection mask
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
        """
        Update the selected_mask based on checked items and close the dialog.

        This method is called when the user clicks the Select button. It updates
        the selected_mask attribute based on which items are checked in the list,
        then closes the dialog with an "accept" result.

        Args:
            index (int): The index of the currently selected item (not used).
        """
        # Update selected_mask based on checked state of each item
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            self.selected_mask[i] = item.checkState() == Qt.Checked
        self.accept()

    def clusterOptions(self, index):
        """
        Create and show a Cluster widget with the selected images.

        This method creates a mask based on which items are checked in the list,
        then creates a Cluster widget with the selected images and displays it.

        Args:
            index (int): The index of the currently selected item (not used).
        """
        # Create mask based on checked state of each item
        mask = np.zeros((self.clusterList.count()))

        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            mask[i] = item.checkState() == Qt.Checked
            # Commented code below was likely used in an earlier version
            #if item.checkState() == Qt.Checked:
            #    self.clusterImgName.append(item.text())
            #    self.clusterImages.append(self.rawImages[i])
            #print(self.clusterImgName)
            #print(self.clusterImages)
            #self.cluster = Cluster(self.clusterImgName, self.clusterImages)

        # Create and show the Cluster widget
        self.cluster = Cluster(None, self._image_set, mask)
        self.close()
        self.cluster.show()

# Configure images for K-means clustering and run K-means
def kmeansCluster(self, index):
    """
    Configure images for K-means clustering.

    This function prepares the selected images for K-means clustering by flattening
    them into a format suitable for the clustering algorithm.

    Note:
        This appears to be a legacy function that may not be fully implemented or
        used in the current version of the application. The Cluster class now handles
        the clustering functionality.

    Args:
        self: The instance containing clusterImages.
        index (int): Index of the image set to use for clustering.

    Returns:
        None
    """
    # Get the images at the specified index
    flatImages = self.clusterImages[index]
    nImages = len(flatImages[0])

    # Reshape each image to a 1D array
    for index in range(len(flatImages[0])):
        flatImg = flatImages[0][index]
        flatImg.reshape([-1], 1)  # Reshape to column vector

    # Prepare input for K-means (note: this doesn't actually run K-means)
    kmeansInput = (flatImg, nImages)
    kmeansInput = np.float32(kmeansInput)  # Convert to float32 for compatibility with clustering algorithms
