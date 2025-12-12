import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QListWidget, QPushButton, QWidget

from cmp_viewer.cluster_widget import Cluster
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
        roiList (QListWidget): Optional list of available cluster masks (ROI) to apply.
        selected_roi_cluster_id (int|None): The chosen ROI cluster id or None for full image.
        selected_roi_mask (NDArray[bool]|None): The chosen ROI mask or None for full image.
    """
    clusterImgName = []
    clusterImages = []

    def __init__(self, fileNameList, image_set: ImageSet, checked=False, *, masks: dict | None = None, names: dict | None = None, **kwargs):
        """
        Initialize the image selection dialog.

        Args:
            fileNameList (list): List of image filenames to display.
            image_set (ImageSet): Set of images that can be selected for clustering.
            checked (bool, optional): Whether all images should be checked by default. Defaults to False.
            masks (Optional[Dict[int, Tuple[NDArray[bool], QColor]]]): Available cluster masks and colors.
            names (Optional[Dict[int, str]]): Display names for cluster ids.
            **kwargs: Additional arguments to pass to the QDialog constructor.
        """
        super().__init__(**kwargs)
        self.cluster = None
        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        self._image_set = image_set
        self._available_masks = masks or {}
        self._names = names or {}

        # Create list items with checkboxes for each image
        for items in fileNameList:
            item = QtWidgets.QListWidgetItem(items)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            check = Qt.Checked if checked else Qt.Unchecked
            item.setCheckState(check)
            self.clusterList.addItem(item)

        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        # Add widgets to layout
        layout.addWidget(QtWidgets.QLabel("Select Images to include in clustering"))
        layout.addWidget(self.clusterList)

        # Optional ROI section
        self.roiList = None
        self.selected_roi_cluster_id = None
        self.selected_roi_mask = None
        if len(self._available_masks) > 0:
            layout.addWidget(QtWidgets.QLabel("Available Cluster Masks (ROI)"))
            self.roiList = QListWidget()
            self.roiList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            # Add custom stylesheet for better visual feedback
            self.roiList.setStyleSheet(
                """
                            QListWidget::item:selected {
                                background-color: #4A90E2;
                                color: white;
                                border: 2px solid #2E5C8A;
                                font-weight: bold;
                            }
                            QListWidget::item:hover {
                                background-color: #16161616;
                            }
                            """
                        )

            # Add "No ROI" option
            no_roi_item = QtWidgets.QListWidgetItem("No ROI (full image)")
            no_roi_item.setData(Qt.UserRole, None)
            self.roiList.addItem(no_roi_item)
            self.roiList.setCurrentItem(no_roi_item)

            # Add one item per cluster mask
            for cid, (mask, color) in self._available_masks.items():
                # Handle both integer cluster IDs and string saved mask IDs
                if isinstance(cid, str):
                    # It's a saved mask with a string key like 'saved_1'
                    name = self._names.get(cid, f"Saved Mask {cid}")
                    item = QtWidgets.QListWidgetItem(name)
                    item.setData(Qt.UserRole, cid)  # Store as string
                else:
                    # It's a live cluster with an integer key
                    name = self._names.get(int(cid), f"Cluster {int(cid)}")
                    item = QtWidgets.QListWidgetItem(name)
                    item.setData(Qt.UserRole, int(cid))  # Store as int

                try:
                    # Use background to show color
                    if isinstance(color, QColor):
                        item.setBackground(color)
                        # Ensure text readable
                        brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
                        item.setForeground(Qt.black if brightness > 128 else Qt.white)
                except Exception:
                    pass
                self.roiList.addItem(item)

            # Helpful hint
            layout.addWidget(QtWidgets.QLabel("Tip: If you choose a mask here, the next clustering will update only that region; other regions are preserved."))
            layout.addWidget(self.roiList)

        # Selection controls
        select_all_button = QPushButton("Select All Images")
        select_all_button.clicked.connect(self.select_all)
        layout.addWidget(select_all_button)

        select_none_button = QPushButton("Select No Images")
        select_none_button.clicked.connect(self.select_none)
        layout.addWidget(select_none_button)

        invert_button = QPushButton("Invert Image Selection")
        invert_button.clicked.connect(self.invert_selection)
        layout.addWidget(invert_button)

        # Select button
        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Select")
        self.button1.clicked.connect(lambda: self.return_results(self.clusterList.currentRow()))
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
        """
        # Update selected_mask based on checked state of each item
        for i in range(self.clusterList.count()):
            item = self.clusterList.item(i)
            self.selected_mask[i] = item.checkState() == Qt.Checked

        # Record selected ROI, if ROI list is present
        if self.roiList is not None:
            # Get the selected items (not just current item)
            selected_items = self.roiList.selectedItems()
            if selected_items and len(selected_items) > 0:
                sel_item = selected_items[0]  # Get first selected item
                cid = sel_item.data(Qt.UserRole)
                if cid is None:
                    self.selected_roi_cluster_id = None
                    self.selected_roi_mask = None
                else:
                    # cid can be either an int (live cluster) or str (saved mask like 'saved_1')
                    self.selected_roi_cluster_id = cid  # Store as-is (int or str)
                    mask_color = self._available_masks.get(cid)  # Look up using the key as-is
                    if mask_color is not None:
                        self.selected_roi_mask = mask_color[0]
                    else:
                        self.selected_roi_mask = None
            else:
                # No items selected, fall back to current item
                if self.roiList.currentItem() is not None:
                    sel_item = self.roiList.currentItem()
                    cid = sel_item.data(Qt.UserRole)
                    if cid is None:
                        self.selected_roi_cluster_id = None
                        self.selected_roi_mask = None
                    else:
                        self.selected_roi_cluster_id = cid
                        mask_color = self._available_masks.get(cid)
                        if mask_color is not None:
                            self.selected_roi_mask = mask_color[0]
                        else:
                            self.selected_roi_mask = None
                else:
                    self.selected_roi_cluster_id = None
                    self.selected_roi_mask = None
        else:
            self.selected_roi_cluster_id = None
            self.selected_roi_mask = None

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
