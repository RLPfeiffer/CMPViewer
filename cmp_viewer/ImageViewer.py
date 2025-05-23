# Filename: ImageViewer.py
# python -m cmp_viewer.imageviewer
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
import sys
import cv2
import os
import glob
from cmp_viewer.rgb import *
from cmp_viewer.clusterImgSelect import *
from cmp_viewer.Cluster import *
import nornir_imageregistration
from cmp_viewer import models
from PIL import Image
import typing
import numpy as np

from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QColor
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
from PyQt5.QtWidgets import QListView
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QSlider, QProgressDialog, QListWidgetItem
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, qRgb
from PyQt5.QtWidgets import QMessageBox
from functools import partial

__version__ = '1.5.2'
__author__ = "RL Pfeiffer & NQN Studios"

class ImageViewerUi(QMainWindow):
    rawImages = []
    fileNameList = []
    r_image = None
    g_image = None
    b_image = None
    _clustered_image = None  # Store clustered image
    _num_labels = None  # Store number of labels
    _color_table = None  # Store color table for saving
    _masks = None  # Store masks and colors for each cluster
    _visible_clusters = set()  # Track which clusters' masks are visible
    _mask_opacity = 100  # Default opacity (0-255)
    last_pixmap = None  # Cache for the last rendered base image
    mask_overlays = {}  # Cache for mask overlays
    last_grayscale_index = None  # Track the last grayscale image index
    last_rgb_state = None  # Track the last RGB state
    overlay_items = {}  # Track overlay items in the scene by cluster ID
    """View Gui"""

    def __init__(self, starting_images_folder=None):
        """View Initializer"""
        super().__init__()

        self._image_set = models.ImageSet()
        self.clusterview = None
        self.imageSelector = None  # Set to None since we're not using it
        self.setWindowTitle('ImageViewer')

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        # Create main horizontal layout
        self.generalLayout = QHBoxLayout()
        self.centralWidget.setLayout(self.generalLayout)

        # Create a vertical layout for the left side controls
        self.leftControlsLayout = QVBoxLayout()

        # Create a widget to hold the left controls layout
        self.leftControlsWidget = QWidget()
        self.leftControlsWidget.setLayout(self.leftControlsLayout)

        # Create a scroll area for the left controls
        self.leftControlsScrollArea = QScrollArea()
        self.leftControlsScrollArea.setWidget(self.leftControlsWidget)
        self.leftControlsScrollArea.setWidgetResizable(True)
        self.leftControlsScrollArea.setFixedWidth(400)  # Fixed width to prevent overlap
        self.leftControlsScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.leftControlsScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add the left controls scroll area to the main layout
        self.generalLayout.addWidget(self.leftControlsScrollArea)

        self.centralWidget.setMinimumSize(1500, 1000)

        self._createDisplay()
        self._createMenuBar()
        self._createViewList()
        self._createIterativeClusteringControls()

        # Removed automatic image loading to ensure no images are open on startup

    def _createMenuBar(self):
        """Create a menubar"""
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)
        fileMenu = menuBar.addMenu('File')
        clusterMenu = menuBar.addMenu('Cluster')

        openAct = QAction('Open Images', self)
        openAct.setShortcut('Ctrl+O')
        openAct.triggered.connect(self.on_open_images_menu)
        fileMenu.addAction(openAct)

        saveAct = QAction('Save Clustered Image', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.triggered.connect(self.save_clustered_image)
        fileMenu.addAction(saveAct)

        closeImagesAct = QAction('Close Images', self)
        closeImagesAct.setShortcut('Ctrl+W')
        closeImagesAct.triggered.connect(self.close_images)
        fileMenu.addAction(closeImagesAct)

        resetViewerAct = QAction('Reset Viewer', self)
        resetViewerAct.setShortcut('Ctrl+R')
        resetViewerAct.triggered.connect(self.reset_viewer)
        fileMenu.addAction(resetViewerAct)

        closeAct = QAction('Close', self)
        closeAct.setShortcut('Ctrl+Q')
        closeAct.triggered.connect(self.close)
        fileMenu.addAction(closeAct)

        selectImagesAct = QAction('Select Images', self)
        selectImagesAct.triggered.connect(self.selectClustImages)
        clusterMenu.addAction(selectImagesAct)

    def _createViewList(self):
        self.ViewList_Box = QtWidgets.QGroupBox('Images')
        # Remove the maximum height restriction to allow the box to expand as needed
        self.ViewList_Layout = QVBoxLayout()
        self.ViewList_Box.setLayout(self.ViewList_Layout)

        # Raw Image Data section
        self.rawImageGroup = QtWidgets.QGroupBox('Raw Image Data')
        self.rawImageGroup.setMinimumHeight(200)  # Set larger minimum height to ensure visibility
        self.rawImageGroup.setVisible(True)  # Ensure the group box is visible
        self.rawLayout = QVBoxLayout()  # Make rawLayout an instance variable
        self.rawImageGroup.setLayout(self.rawLayout)
        self.ViewList_Layout.addWidget(self.rawImageGroup)

        # Initialize ImportLayout for radio buttons inside rawLayout
        self.ImportLayout = QVBoxLayout()
        self.ImportLayout.setSpacing(10)  # Add spacing between radio button rows
        self.rawLayout.addLayout(self.ImportLayout)

        # Initialize button groups for radio buttons
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        # Add to the left controls layout instead of directly to generalLayout
        self.leftControlsLayout.addWidget(self.ViewList_Box)

    def _createDisplay(self):
        self.display = QScrollArea()
        self.displayView = QGraphicsView()
        self.displayImage = QGraphicsScene()

        self.display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setWidgetResizable(True)

        self.displayView.setScene(self.displayImage)
        self.display.setWidget(self.displayView)
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(1100, 1000)

        # Add the display to the general layout with a stretch factor to give it priority
        self.generalLayout.addWidget(self.display, 1)  # Stretch factor of 1 makes it expand

    def _createIterativeClusteringControls(self):
        self.iterativeClusterBox = QtWidgets.QGroupBox('Iterative Clustering')
        self.iterativeClusterLayout = QVBoxLayout()
        self.iterativeClusterBox.setLayout(self.iterativeClusterLayout)

        self.clusterSelectCombo = QComboBox()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.iterativeClusterLayout.addWidget(self.clusterSelectCombo)

        self.subClustersInput = QLineEdit()
        self.subClustersInput.setPlaceholderText("Number of sub-clusters")
        self.iterativeClusterLayout.addWidget(self.subClustersInput)

        self.iterativeClusterButton = QPushButton("Run Iterative Clustering")
        self.iterativeClusterButton.clicked.connect(self.run_iterative_clustering)
        self.iterativeClusterLayout.addWidget(self.iterativeClusterButton)

        self.clusterVisibilityList = QListWidget()
        self.clusterVisibilityList.setMinimumHeight(100)
        self.iterativeClusterLayout.addWidget(QLabel("Cluster Mask Visibility"))
        self.iterativeClusterLayout.addWidget(self.clusterVisibilityList)

        self.opacitySlider = QSlider(Qt.Horizontal)
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(255)
        self.opacitySlider.setValue(self._mask_opacity)
        self.opacitySlider.setTickPosition(QSlider.TicksBelow)
        self.opacitySlider.setTickInterval(25)
        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)
        self.iterativeClusterLayout.addWidget(QLabel("Mask Opacity"))
        self.iterativeClusterLayout.addWidget(self.opacitySlider)

        self.exportFormatCombo = QComboBox()
        self.exportFormatCombo.addItems(["PNG", "BMP", "TIFF"])
        self.exportFormatCombo.setCurrentText("PNG")
        self.iterativeClusterLayout.addWidget(QLabel("Export Format"))
        self.iterativeClusterLayout.addWidget(self.exportFormatCombo)

        self.exportMasksButton = QPushButton("Export Cluster Masks")
        self.exportMasksButton.clicked.connect(self.export_cluster_masks)
        self.iterativeClusterLayout.addWidget(self.exportMasksButton)

        self.undoButton = QPushButton("Undo Clustering")
        self.undoButton.clicked.connect(self.undo_clustering)
        self.iterativeClusterLayout.addWidget(self.undoButton)

        # Add to the left controls layout instead of directly to generalLayout
        self.leftControlsLayout.addWidget(self.iterativeClusterBox)

    def chooseGrayscaleImage(self, index):
        if self.last_grayscale_index == index and self.last_pixmap is not None:
            print("No change in grayscale image, reusing last pixmap")
            return

        self.last_grayscale_index = index
        self.last_rgb_state = None

        img_array = self.rawImages[index]
        gray1D = img_array.tobytes()
        qImg = QImage(gray1D, img_array.shape[1], img_array.shape[0], QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qImg).scaled(2000, 5000, Qt.KeepAspectRatio)
        self.displayImage.clear()
        self.overlay_items.clear()  # Clear overlay items since scene is cleared
        self.displayImage.addPixmap(pixmap)
        self.last_pixmap = pixmap
        self.adjustSize()
        print(f"Rendered grayscale image at index {index}")

    def chooseRedImage(self, index):
        self.r_image = self.rawImages[index]
        current_rgb_state = (id(self.r_image), id(self.g_image), id(self.b_image))
        if self.last_rgb_state == current_rgb_state and self.last_pixmap is not None:
            print("No change in RGB state, reusing last pixmap")
            return

        self.last_rgb_state = current_rgb_state
        self.last_grayscale_index = None

        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite).scaled(2000, 5000, Qt.KeepAspectRatio)
        self.displayImage.clear()
        self.overlay_items.clear()  # Clear overlay items since scene is cleared
        self.displayImage.addPixmap(pixmap)
        self.last_pixmap = pixmap
        self.adjustSize()
        print("Rendered red channel")

    def chooseGreenImage(self, index):
        self.g_image = self.rawImages[index]
        current_rgb_state = (id(self.r_image), id(self.g_image), id(self.b_image))
        if self.last_rgb_state == current_rgb_state and self.last_pixmap is not None:
            print("No change in RGB state, reusing last pixmap")
            return

        self.last_rgb_state = current_rgb_state
        self.last_grayscale_index = None

        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite).scaled(2000, 5000, Qt.KeepAspectRatio)
        self.displayImage.clear()
        self.overlay_items.clear()  # Clear overlay items since scene is cleared
        self.displayImage.addPixmap(pixmap)
        self.last_pixmap = pixmap
        self.adjustSize()
        print("Rendered green channel")

    def chooseBlueImage(self, index):
        self.b_image = self.rawImages[index]
        current_rgb_state = (id(self.r_image), id(self.g_image), id(self.b_image))
        if self.last_rgb_state == current_rgb_state and self.last_pixmap is not None:
            print("No change in RGB state, reusing last pixmap")
            return

        self.last_rgb_state = current_rgb_state
        self.last_grayscale_index = None

        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite).scaled(2000, 5000, Qt.KeepAspectRatio)
        self.displayImage.clear()
        self.overlay_items.clear()  # Clear overlay items since scene is cleared
        self.displayImage.addPixmap(pixmap)
        self.last_pixmap = pixmap
        self.adjustSize()
        print("Rendered blue channel")

    def on_open_images_menu(self):
        results = QFileDialog.getOpenFileNames(self, self.tr("Select image(s) to open"))
        self.open_images(results[0])

    def open_images(self, filenames: list[str]):
        # Clear existing ImportLayout
        while self.ImportLayout.count():
            item = self.ImportLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Clear button groups
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        # Populate data structures
        self.fileNameList.clear()
        self.rawImages.clear()
        for index, filename in enumerate(filenames):
            basefileName = os.path.basename(filename)
            simpleName = os.path.splitext(basefileName)[0]
            self.fileNameList.append(simpleName)
            self.rawImages.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

        # Process images for display
        for index, filename in enumerate(filenames):
            self.importImageWrapper(filename)
            self.colorRBs(filename, index)

        # Default to first image
        if filenames and self.column1.buttons() and len(self.column1.buttons()) > 0:
            # Select the first image's grayscale radio button
            self.column1.buttons()[0].setChecked(True)

    def importImageWrapper(self, fileName):
        '''
        Imports images into UI
        :param str fileName: Filename designated by openImages
        :return: viewable image with color select radio buttons
        :rtype: numpy nd array
        '''
        image = nornir_imageregistration.LoadImage(fileName)
        image_float = image.astype(np.float32)
        self._image_set.add_image(image_float)

    def colorRBs(self, fileName, index):
        print(f"Creating radio buttons for image {index}: {fileName}")

        row = QtWidgets.QGroupBox()
        row.setMinimumHeight(30)  # Set minimum height to ensure visibility
        rowLayout = QHBoxLayout()

        basefileName = os.path.basename(fileName)
        simpleName = os.path.splitext(basefileName)[0]
        rowLayout.addWidget(QLabel(simpleName))

        grayRadioButton = QRadioButton('gray')
        grayRadioButton.toggled.connect(lambda: self.chooseGrayscaleImage(index))
        rowLayout.addWidget(grayRadioButton)
        self.column1.addButton(grayRadioButton)
        print(f"Added gray radio button for image {index}")

        redRadioButton = QRadioButton("R")
        redRadioButton.toggled.connect(lambda: self.chooseRedImage(index))
        rowLayout.addWidget(redRadioButton)
        self.redRBlist.addButton(redRadioButton)

        greenRadioButton = QRadioButton("G")
        greenRadioButton.toggled.connect(lambda: self.chooseGreenImage(index))
        rowLayout.addWidget(greenRadioButton)
        self.greenRBlist.addButton(greenRadioButton)

        blueRadioButton = QRadioButton("B")
        blueRadioButton.toggled.connect(lambda: self.chooseBlueImage(index))
        rowLayout.addWidget(blueRadioButton)
        self.blueRBlist.addButton(blueRadioButton)

        row.setLayout(rowLayout)
        self.ImportLayout.addWidget(row)
        print(f"Added row with radio buttons to ImportLayout for image {index}")

        # Force update to ensure the radio buttons are displayed
        self.ImportLayout.update()
        self.rawLayout.update()
        self.rawImageGroup.update()

        # Print the number of buttons in each group for debugging
        print(f"Number of buttons in column1: {len(self.column1.buttons())}")
        print(f"Number of buttons in redRBlist: {len(self.redRBlist.buttons())}")
        print(f"Number of buttons in greenRBlist: {len(self.greenRBlist.buttons())}")
        print(f"Number of buttons in blueRBlist: {len(self.blueRBlist.buttons())}")

    def save_clustered_image(self):
        if self._clustered_image is None:
            QtWidgets.QMessageBox.warning(self, "No Clustered Image", "No clustered image available to save.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Clustered Image",
            "",
            "BMP Files (*.bmp);;PNG Files (*.png);;All Files (*)"
        )

        if file_name:
            if not file_name.lower().endswith(('.bmp', '.png')):
                file_name += '.bmp'
            self._clustered_image.save(file_name)
            QtWidgets.QMessageBox.information(self, "Success", f"Clustered image saved to {file_name}")

    def close_images(self, show_message=True):
        """
        Close all open images and reset the viewer.

        This method clears all image data, resets UI elements, and optionally
        shows a success message.

        Args:
            show_message (bool, optional): Whether to show a success message. 
                                          Defaults to True.
        """
        try:
            self.opacitySlider.valueChanged.disconnect()
        except Exception:
            pass

        self.rawImages.clear()
        self.fileNameList.clear()
        self.r_image = None
        self.g_image = None
        self.b_image = None
        self._clustered_image = None
        self._num_labels = None
        self._color_table = None
        self._masks = None
        self._visible_clusters.clear()
        self._mask_opacity = 100
        self.last_pixmap = None
        self.mask_overlays.clear()
        self.last_grayscale_index = None
        self.last_rgb_state = None
        self.overlay_items.clear()
        if self.clusterview:
            self.clusterview.undo_stack.clear()
        self._image_set = models.ImageSet()
        self.displayImage.clear()

        self.clusterSelectCombo.clear()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.subClustersInput.clear()
        self.clusterVisibilityList.clear()
        self.opacitySlider.setValue(self._mask_opacity)
        self.exportFormatCombo.setCurrentText("PNG")

        while self.ImportLayout.count():
            item = self.ImportLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset button groups
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)

        if show_message:
            QMessageBox.information(self, "Success", "All images have been closed.")

    def closeEvent(self, event):
        """
        Handle the window close event.

        This method is called automatically when the window is closed.
        It ensures all images are cleared before the application exits.

        Args:
            event (QCloseEvent): The close event.
        """
        # Clear all images before closing, but don't show a message
        self.close_images(show_message=False)
        # Accept the close event to allow the window to close
        event.accept()

    def reset_viewer(self):
        try:
            self.opacitySlider.valueChanged.disconnect()
        except Exception:
            pass

        self.rawImages.clear()
        self.fileNameList.clear()
        self.r_image = None
        self.g_image = None
        self.b_image = None
        self._clustered_image = None
        self._num_labels = None
        self._color_table = None
        self._masks = None
        self._visible_clusters.clear()
        self._mask_opacity = 100
        self.last_pixmap = None
        self.mask_overlays.clear()
        self.last_grayscale_index = None
        self.last_rgb_state = None
        self.overlay_items.clear()
        if self.clusterview:
            self.clusterview.undo_stack.clear()
        self._image_set = models.ImageSet()
        self.displayImage.clear()

        # Clear cluster-related UI elements
        self.clusterSelectCombo.clear()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.subClustersInput.clear()
        self.clusterVisibilityList.clear()
        self.opacitySlider.setValue(self._mask_opacity)
        self.exportFormatCombo.setCurrentText("PNG")

        # Clear the ImportLayout (radio buttons)
        while self.ImportLayout.count():
            item = self.ImportLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset button groups
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        # Remove widgets from generalLayout but keep the layout structure
        if self.leftControlsScrollArea in self.generalLayout:
            self.generalLayout.removeWidget(self.leftControlsScrollArea)
        if self.display in self.generalLayout:
            self.generalLayout.removeWidget(self.display)

        # Recreate the display
        self._createDisplay()

        # Re-add the left controls scroll area to the main layout
        self.generalLayout.insertWidget(0, self.leftControlsScrollArea)

        # Reconnect signals
        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)

        QMessageBox.information(self, "Success", "Viewer has been reset.")

    def selectClustImages(self):
        try:
            num_images = self._image_set.num_images
            if num_images == 0:
                QMessageBox.warning(self, "No Images Loaded", "Please load images before selecting for clustering.")
                return
        except AttributeError:
            QMessageBox.warning(self, "Image Set Error", "The image set is not properly initialized. Please load images first.")
            return

        select_dlg = ImageSelectDlg(self.fileNameList, self._image_set)
        select_dlg.setModal(True)

        select_dlg.exec_()

        self.clusterview = Cluster(self.fileNameList, self._image_set, select_dlg.selected_mask, self.on_cluster_callback)
        self.clusterview.show()

    def on_cluster_callback(self, labels: NDArray[int], settings: typing.Any):
        if self.clusterview is None:
            print("Cannot process cluster callback: Clusterview is None.")
            return

        # Use Cluster class to create the label image
        pillow_img = self.clusterview.create_label_image(labels, len(np.unique(labels)))
        if isinstance(settings, KMeansSettings) or isinstance(settings, ISODATASettings):
            self._masks = self.clusterview.masks
            self.clusterSelectCombo.clear()
            self.clusterSelectCombo.addItem("Select Cluster")
            for cluster_id in np.unique(labels):
                self.clusterSelectCombo.addItem(f"Cluster {cluster_id}")
            self.update_cluster_visibility_list(labels)
            self.show_label_image(pillow_img, len(np.unique(labels)))

    def update_cluster_visibility_list(self, labels: NDArray[int]):
        current_clusters = {int(item.text().split()[-1]) for i in range(self.clusterVisibilityList.count())
                           for item in [self.clusterVisibilityList.item(i)]}
        new_clusters = set(np.unique(labels))

        # Add new clusters
        for cluster_id in new_clusters - current_clusters:
            item = QListWidgetItem(f"Cluster {cluster_id}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.clusterVisibilityList.addItem(item)

        # Remove obsolete clusters
        for i in reversed(range(self.clusterVisibilityList.count())):
            item = self.clusterVisibilityList.item(i)
            cluster_id = int(item.text().split()[-1])
            if cluster_id not in new_clusters:
                self.clusterVisibilityList.takeItem(i)
                self._visible_clusters.discard(cluster_id)

        self.clusterVisibilityList.itemChanged.connect(self.toggle_cluster_visibility)
        print(f"Updated cluster visibility list with {len(new_clusters)} clusters")

    def toggle_cluster_visibility(self, item):
        cluster_id = int(item.text().split()[-1])
        print(f"Toggling visibility for cluster {cluster_id}, check state: {item.checkState()}")
        if item.checkState() == Qt.Checked:
            self._visible_clusters.add(cluster_id)
        else:
            self._visible_clusters.discard(cluster_id)
        self.show_label_image(self._clustered_image, self._num_labels)

    def run_iterative_clustering(self):
        if self.clusterview is None or self._masks is None:
            QMessageBox.warning(self, "No Clustering Data", "Please run initial clustering first.")
            return

        selected_cluster = self.clusterSelectCombo.currentText()
        if selected_cluster == "Select Cluster":
            QMessageBox.warning(self, "Invalid Selection", "Please select a cluster to refine.")
            return

        cluster_id = int(selected_cluster.split()[-1])
        mask, _ = self._masks.get(cluster_id, (None, None))
        if mask is None:
            QMessageBox.warning(self, "Invalid Mask", "Selected cluster mask is not available.")
            return

        try:
            n_sub_clusters = int(self.subClustersInput.text())
            if n_sub_clusters < 2:
                raise ValueError("Number of sub-clusters must be at least 2.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of sub-clusters (at least 2).")
            return

        new_labels, new_settings = self.clusterview.cluster_on_mask(mask, n_sub_clusters)
        if new_labels is None:
            QMessageBox.warning(self, "No Pixels", "No pixels in the selected mask region for clustering.")
            return

        self.on_cluster_callback(new_labels, new_settings)

    def undo_clustering(self):
        if self.clusterview and self.clusterview.undo_clustering():
            self.on_cluster_callback(self.clusterview.labels, KMeansSettings(n_clusters=len(np.unique(self.clusterview.labels)), init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42))
        else:
            QMessageBox.warning(self, "No Undo", "No previous clustering state available to undo.")

    def update_mask_opacity(self, value):
        self._mask_opacity = value
        # Clear mask overlay cache to force recompute with new opacity
        self.mask_overlays.clear()
        if self._clustered_image is not None and self._num_labels is not None:
            self.show_label_image(self._clustered_image, self._num_labels)
        else:
            print("Cannot update mask opacity: Clustered image or number of labels is not set.")

    def export_cluster_masks(self):
        if self.clusterview is None:
            QMessageBox.warning(self, "No Clustering Data", "Please run clustering first.")
            return

        if self._masks is None:
            QMessageBox.warning(self, "No Masks Available", "Please run clustering to generate masks before exporting.")
            return

        if not self._visible_clusters:
            QMessageBox.warning(self, "No Clusters Selected", "Please select at least one cluster for export by checking the visibility list.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Cluster Masks")
        if not output_dir:
            return

        file_format = self.exportFormatCombo.currentText().lower()

        progress = QProgressDialog("Exporting cluster masks...", "Cancel", 0, len(self._visible_clusters), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(1000)

        for i, cluster_id in enumerate(self._visible_clusters):
            if progress.wasCanceled():
                break

            # Use Cluster class to export the mask
            output_path = os.path.join(output_dir, f"cluster_{cluster_id}_mask")
            success = self.clusterview.export_cluster_mask(cluster_id, output_path, file_format)

            if not success:
                print(f"Failed to export mask for cluster {cluster_id}")

            progress.setValue(i + 1)

        if not progress.wasCanceled():
            QMessageBox.information(self, "Success", f"Cluster masks exported to {output_dir}")
        else:
            QMessageBox.information(self, "Export Canceled", "Export process was canceled.")

    def show_label_image(self, img, num_labels: int):
        if img is None or num_labels is None:
            print("Cannot show label image: Image or num_labels is None.")
            return

        if self.clusterview is None:
            print("Cannot show label image: Clusterview is None.")
            return

        base_image_changed = self._clustered_image != img or self._num_labels != num_labels
        if base_image_changed:
            self._clustered_image = img
            self._num_labels = num_labels

            # Use Cluster class to prepare the image and get the color table
            prepared_img, self._color_table = self.clusterview.prepare_label_image_for_display(img, num_labels)
            self._clustered_image = prepared_img

            # Create QImage from the prepared image
            qimage = QImage(self._clustered_image.tobytes(), self._clustered_image.size[0], 
                           self._clustered_image.size[1], QImage.Format_Indexed8)
            qimage.setColorCount(num_labels)
            qimage.setColorTable(self._color_table)

            # Create and display pixmap
            pixmap = QPixmap.fromImage(qimage).scaled(2000, 5000, Qt.KeepAspectRatio)
            self.last_pixmap = pixmap
            self.last_grayscale_index = None
            self.last_rgb_state = None
            self.displayImage.clear()
            self.overlay_items.clear()  # Clear overlay items since scene is cleared
            self.displayImage.addPixmap(pixmap)
            print("Rendered clustered image")

        # Remove overlays for deselected clusters
        print(f"Before removal: overlay_items = {self.overlay_items.keys()}")
        clusters_to_remove = set(self.overlay_items.keys()) - self._visible_clusters
        print(f"Clusters to remove: {clusters_to_remove}")
        for cluster_id in clusters_to_remove:
            if cluster_id in self.overlay_items:
                item = self.overlay_items[cluster_id]
                try:
                    if item.scene() == self.displayImage:
                        self.displayImage.removeItem(item)
                        print(f"Removed overlay for cluster {cluster_id}")
                    else:
                        print(f"Skipping removal of cluster {cluster_id}: item is not in the current scene")
                except Exception as e:
                    print(f"Error removing overlay for cluster {cluster_id}: {e}")
                finally:
                    del self.overlay_items[cluster_id]
        print(f"After removal: overlay_items = {self.overlay_items.keys()}")

        # Handle mask overlays for visible clusters
        if self._masks and self._visible_clusters:
            first_mask, _ = next(iter(self._masks.values()))
            height, width = first_mask.shape

            # Calculate optimal scale factor
            if self.clusterview is not None:
                scale_factor = self.clusterview.calculate_optimal_scale_factor(height, width)
            else:
                # Fallback if clusterview is None
                max_pixels = 500000
                scale_factor = np.sqrt(max_pixels / (height * width)) if height * width > max_pixels else 1.0

            if scale_factor < 1.0:
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
            else:
                new_height, new_width = height, width

            for cluster_id in self._visible_clusters:
                cache_key = (cluster_id, self._mask_opacity, new_width, new_height)
                if cache_key in self.mask_overlays:
                    overlay_pixmap = self.mask_overlays[cache_key]
                    if cluster_id not in self.overlay_items:
                        item = self.displayImage.addPixmap(overlay_pixmap)
                        self.overlay_items[cluster_id] = item
                        print(f"Reused cached overlay for cluster {cluster_id}")
                    continue

                mask, color = self._masks.get(cluster_id, (None, None))
                if mask is None:
                    continue

                # Create mask overlay
                if self.clusterview is not None:
                    overlay = self.clusterview.create_mask_overlay(
                        mask, color, self._mask_opacity, 
                        target_width=new_width, target_height=new_height
                    )
                else:
                    # Fallback if clusterview is None
                    if new_width != width or new_height != height:
                        mask_small = cv2.resize(mask.astype(np.uint8), (new_width, new_height), 
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask_small = mask

                    overlay = QImage(new_width, new_height, QImage.Format_ARGB32)
                    overlay.fill(Qt.transparent)

                    mask_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                    mask_data[mask_small, 0] = color.red()
                    mask_data[mask_small, 1] = color.green()
                    mask_data[mask_small, 2] = color.blue()
                    mask_data[mask_small, 3] = self._mask_opacity

                    overlay_data = mask_data.tobytes()
                    overlay = QImage(overlay_data, new_width, new_height, QImage.Format_ARGB32)

                # Create and cache pixmap
                overlay_pixmap = QPixmap.fromImage(overlay).scaled(2000, 5000, Qt.KeepAspectRatio)
                self.mask_overlays[cache_key] = overlay_pixmap

                # Remove existing overlay if present
                if cluster_id in self.overlay_items:
                    try:
                        if self.overlay_items[cluster_id].scene() == self.displayImage:
                            self.displayImage.removeItem(self.overlay_items[cluster_id])
                        else:
                            print(f"Skipping removal of existing overlay for cluster {cluster_id}: item is not in the current scene")
                    except Exception as e:
                        print(f"Error removing existing overlay for cluster {cluster_id}: {e}")

                # Add new overlay
                item = self.displayImage.addPixmap(overlay_pixmap)
                self.overlay_items[cluster_id] = item
                print(f"Computed and cached overlay for cluster {cluster_id}")

        self.adjustSize()


# Client code
def main():
    starting_images_folder = os.environ['DEBUG_IMAGES_FOLDER'] if 'DEBUG_IMAGES_FOLDER' in os.environ else None
    CMPViewer = QApplication(sys.argv)
    view = ImageViewerUi(starting_images_folder)
    view.show()
    sys.exit(CMPViewer.exec_())

if __name__ == '__main__':
    main()
