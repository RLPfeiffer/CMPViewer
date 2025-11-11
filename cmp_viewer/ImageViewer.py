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
import datetime
from cmp_viewer import models
from PIL import Image
import typing
import numpy as np
import re
import csv

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
from PyQt5.QtWidgets import QSlider, QProgressDialog, QListWidgetItem, QColorDialog, QMenu, QInputDialog
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
    _selected_clusters = set()  # Track which clusters are selected for reclustering
    _mask_opacity = 100  # Default opacity (0-255)
    last_pixmap = None  # Cache for the last rendered base image
    mask_overlays = {}  # Cache for mask overlays
    last_grayscale_index = None  # Track the last grayscale image index
    last_rgb_state = None  # Track the last RGB state
    overlay_items = {}  # Track overlay items in the scene by cluster ID
    _cluster_names = {}  # Optional custom names for clusters (cluster_id -> name)
    _last_selected_roi_cluster_id = None  # Tracks the ROI cluster id used for the last clustering (if any)
    _last_selected_roi_name = None  # Tracks the ROI cluster display name used for the last clustering (if any)

    # Saved/Protected masks (session-only)
    _saved_masks = {}  # saved_id -> (mask: np.ndarray[bool], color: QColor, name: str)
    _saved_visible = set()  # saved_id items currently visible
    _next_saved_id = 1  # incrementing ID for saved masks
    saved_mask_overlays = {}  # cache for saved mask overlays
    saved_overlay_items = {}  # graphics items for saved overlays
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
        self.iterativeClusterBox = QtWidgets.QGroupBox('Cluster Masks')
        self.iterativeClusterLayout = QVBoxLayout()
        self.iterativeClusterBox.setLayout(self.iterativeClusterLayout)


        self.clusterVisibilityList = QListWidget()
        self.clusterVisibilityList.setMinimumHeight(100)
        self.clusterVisibilityList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clusterVisibilityList.customContextMenuRequested.connect(self.show_cluster_context_menu)
        self.iterativeClusterLayout.addWidget(QLabel("Cluster Mask Visibility (Live)"))
        self.iterativeClusterLayout.addWidget(self.clusterVisibilityList)

        # Saved Masks panel
        self.savedMasksList = QListWidget()
        self.savedMasksList.setMinimumHeight(100)
        self.savedMasksList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.savedMasksList.customContextMenuRequested.connect(self.show_saved_context_menu)
        self.savedMasksList.itemChanged.connect(self.toggle_saved_visibility)
        self.iterativeClusterLayout.addWidget(QLabel("Saved Masks (Protected)"))
        self.iterativeClusterLayout.addWidget(self.savedMasksList)

        # Merge Selected Clusters button (restored)
        self.mergeClusterButton = QPushButton("Merge Selected Clusters")
        self.mergeClusterButton.clicked.connect(self.merge_selected_clusters)
        self.iterativeClusterLayout.addWidget(self.mergeClusterButton)

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

        self.exportSavedMasksButton = QPushButton("Export Saved Masks")
        self.exportSavedMasksButton.clicked.connect(self.export_saved_masks)
        self.iterativeClusterLayout.addWidget(self.exportSavedMasksButton)

        # New: Export intensity stats under a Saved Mask
        self.exportIntensityStatsButton = QPushButton("Export Intensity Stats (Saved Mask)")
        self.exportIntensityStatsButton.clicked.connect(self.export_intensity_stats)
        self.iterativeClusterLayout.addWidget(self.exportIntensityStatsButton)

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
        self._selected_clusters.clear()
        self._mask_opacity = 100
        self.last_pixmap = None
        self.mask_overlays.clear()
        self.last_grayscale_index = None
        self.last_rgb_state = None
        self.overlay_items.clear()
        self._cluster_names.clear()
        # Clear saved masks state
        self._saved_masks.clear()
        self._saved_visible.clear()
        self.saved_mask_overlays.clear()
        self.saved_overlay_items.clear()
        if self.clusterview:
            self.clusterview.undo_stack.clear()
        self._image_set = models.ImageSet()
        self.displayImage.clear()

        self.clusterVisibilityList.clear()
        if hasattr(self, 'savedMasksList'):
            self.savedMasksList.clear()
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
        self._selected_clusters.clear()
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
        self._cluster_names.clear()
        # Clear saved masks state
        self._saved_masks.clear()
        self._saved_visible.clear()
        self.saved_mask_overlays.clear()
        self.saved_overlay_items.clear()

        # Clear cluster-related UI elements
        self.clusterVisibilityList.clear()
        if hasattr(self, 'savedMasksList'):
            self.savedMasksList.clear()

        # Reset slider and combo box
        self.opacitySlider.setValue(self._mask_opacity)
        self.exportFormatCombo.setCurrentText("PNG")

        # Clear ImportLayout
        while self.ImportLayout.count():
            item = self.ImportLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset button groups
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        # Reconnect slider
        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)

        QMessageBox.information(self, "Success", "Viewer has been reset.")

    def selectClustImages(self):
        """
        Open a dialog to select images and optionally a spatial ROI for clustering.

        Creates an ImageSelectDlg dialog that allows the user to select which images
        to include in clustering and optionally select a cluster mask as a Region of Interest (ROI).
        The selected images and ROI are then used to create a Cluster widget.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=== selectClustImages called ===")
        if self._image_set.num_images == 0:
            QtWidgets.QMessageBox.warning(self, "No Images",
                                          "No images are currently loaded. Please open images first.")
            return

        # Combine live masks and saved masks for ROI selection
        all_masks = {}
        all_names = {}

        """
        # Add live cluster masks if they exist
        if self._masks is not None:
            logger.info(f"Adding {len(self._masks)} live masks to dialog")
            for cluster_id, (mask, color) in self._masks.items():
                all_masks[cluster_id] = (mask, color)
                all_names[cluster_id] = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        """
        # Add saved masks if they exist
        if self._saved_masks:
            logger.info(f"Adding {len(self._saved_masks)} saved masks to dialog")
            for saved_id, (mask, color, name) in self._saved_masks.items():
                # Use negative IDs for saved masks to avoid collision with live clusters
                mask_key = f"saved_{saved_id}"
                all_masks[mask_key] = (mask, color)
                all_names[mask_key] = f"[Saved] {name}"

        logger.info(f"Total masks available in dialog: {len(all_masks)}")

        # Pass combined masks and names to the selection dialog
        select_dlg = ImageSelectDlg(
            self.fileNameList,
            self._image_set,
            checked=True,
            masks=all_masks,
            names=all_names
        )

        result = select_dlg.exec_()
        if result != QtWidgets.QDialog.Accepted:
            logger.info("Dialog canceled by user")
            return

        # Extract the selected mask (which images to cluster)
        selected_mask = select_dlg.selected_mask
        logger.info(f"Selected {np.sum(selected_mask)} images for clustering")

        # Extract the selected ROI information
        roi_cluster_id = select_dlg.selected_roi_cluster_id
        roi_mask = select_dlg.selected_roi_mask
        roi_name = None

        logger.info(f"ROI cluster ID from dialog: {roi_cluster_id}")
        logger.info(
            f"ROI mask from dialog: {type(roi_mask)}, shape: {roi_mask.shape if roi_mask is not None else 'None'}")

        # Get the name for the selected ROI
        if roi_cluster_id is not None:
            if isinstance(roi_cluster_id, str) and roi_cluster_id.startswith("saved_"):
                # It's a saved mask
                saved_id = int(roi_cluster_id.split("_")[1])
                roi_name = all_names.get(roi_cluster_id, f"Saved Mask {saved_id}")
            else:
                # It's a live cluster
                roi_name = self._cluster_names.get(roi_cluster_id, f"Cluster {roi_cluster_id}")

            logger.info(f"Selected ROI: {roi_name}")
        else:
            logger.info("No ROI selected (full image clustering)")

        # Get base labels and masks for ROI clustering
        base_labels = None
        base_masks = None

        if self.clusterview is not None:
            base_labels = self.clusterview.labels
            base_masks = self._masks
            logger.info(
                f"Using existing cluster view - labels shape: {base_labels.shape if base_labels is not None else 'None'}")
            logger.info(f"Base masks count: {len(base_masks) if base_masks is not None else 0}")
        else:
            logger.info("No existing cluster view - first clustering run")

        # Create Cluster widget with all the information
        logger.info(f"Creating Cluster widget with spatial_roi: {roi_mask is not None}")
        self.clusterview = Cluster(
            self.fileNameList,
            self._image_set,
            selected_mask,
            self.on_cluster_callback,
            base_labels=base_labels,
            base_masks=base_masks,
            spatial_roi=roi_mask  # Pass the ROI mask here!
        )

        # Store the ROI information for reference
        self._last_selected_roi_cluster_id = roi_cluster_id
        self._last_selected_roi_name = roi_name

        logger.info(f"Cluster widget created. Spatial ROI in Cluster: {self.clusterview._spatial_roi is not None}")
        if self.clusterview._spatial_roi is not None:
            logger.info(
                f"Spatial ROI shape: {self.clusterview._spatial_roi.shape}, True pixels: {np.sum(self.clusterview._spatial_roi)}")

        self.clusterview.show()
        logger.info("=== selectClustImages completed ===")


    def on_cluster_callback(self, labels: NDArray[int], settings: typing.Any):
        if self.clusterview is None:
            print("Cannot process cluster callback: Clusterview is None.")
            return

        # Use Cluster class to create the label image
        pillow_img = self.clusterview.create_label_image(labels, len(np.unique(labels)))

        # Only proceed with mask/name updates if we recognize the settings type
        if isinstance(settings, KMeansSettings) or isinstance(settings, ISODATASettings):
            # Determine if this was ROI-based or full-image clustering
            # Use the Cluster widget's flag to avoid misclassifying ISODATA (which currently ignores ROI)
            is_roi_run = bool(getattr(self.clusterview, '_last_run_was_roi', False))
            roi_id = getattr(self, '_last_selected_roi_cluster_id', None)

            # Previous and new cluster id sets
            if is_roi_run:
                prev_ids = set(self._masks.keys()) if isinstance(self._masks, dict) else set()
            else:
                # Full-image run: treat as fresh result; clear previous names to avoid mismatches
                self._cluster_names.clear()
                prev_ids = set()

            new_ids = set(int(c) for c in np.unique(labels))

            # Update masks from Cluster
            self._masks = self.clusterview.masks

            # Preserve existing names for any cluster ids that persist (ROI runs only)
            # and create names only for genuinely new ids
            created_ids = [cid for cid in new_ids if cid not in prev_ids]

            # Determine base name context if this was ROI-based reclustering
            base_name = None
            if is_roi_run:
                # Prefer the captured ROI display name for stability
                base_name = getattr(self, '_last_selected_roi_name', None)
                if not base_name and roi_id is not None:
                    base_name = self._cluster_names.get(int(roi_id), f"Cluster {int(roi_id)}")

            # Build an algorithm label and proposed name format per new convention
            if isinstance(settings, KMeansSettings):
                algo_label = "Kmeans"
            else:
                algo_label = "ISODATA"

            # Assign readable names to new cluster ids, avoiding overwrite; numbering uses actual cluster id
            for cid in sorted(created_ids):
                if cid in self._cluster_names:
                    continue  # respect any pre-existing name
                if base_name:
                    proposed = f"{algo_label}({base_name}) Cluster {cid}"
                else:
                    proposed = f"{algo_label} Cluster {cid}"

                # Ensure uniqueness among current names
                unique_name = proposed
                bump = 2
                existing_names = set(self._cluster_names.values())
                while unique_name in existing_names:
                    unique_name = f"{proposed} [{bump}]"
                    bump += 1

                self._cluster_names[int(cid)] = unique_name

            # Clear the ROI tracker after use
            self._last_selected_roi_cluster_id = None

            # Update UI
            self.update_cluster_visibility_list(labels)
            self.show_label_image(pillow_img, len(np.unique(labels)))

    def update_cluster_visibility_list(self, labels: NDArray[int]):
        current_clusters = {self.clusterVisibilityList.item(i).data(Qt.UserRole) for i in range(self.clusterVisibilityList.count())}
        new_clusters = set(np.unique(labels))

        # Add new clusters
        for cluster_id in new_clusters - current_clusters:
            display_name = self._cluster_names.get(int(cluster_id), f"Cluster {int(cluster_id)}")
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, int(cluster_id))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)

            # Set the background color to match the cluster color
            if self._masks and int(cluster_id) in self._masks:
                _, color = self._masks.get(int(cluster_id))
                if color:
                    # Set background color to match cluster color
                    item.setBackground(color)

                    # Set text color to black or white depending on the brightness of the background
                    # Using the formula: brightness = 0.299*R + 0.587*G + 0.114*B
                    brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
                    text_color = Qt.black if brightness > 128 else Qt.white
                    item.setForeground(text_color)

            self.clusterVisibilityList.addItem(item)

        # Remove obsolete clusters
        for i in reversed(range(self.clusterVisibilityList.count())):
            item = self.clusterVisibilityList.item(i)
            cluster_id = item.data(Qt.UserRole)
            if cluster_id not in new_clusters:
                self.clusterVisibilityList.takeItem(i)
                self._visible_clusters.discard(int(cluster_id))

        self.clusterVisibilityList.itemChanged.connect(self.toggle_cluster_visibility)
        print(f"Updated cluster visibility list with {len(new_clusters)} clusters")

    def show_cluster_context_menu(self, position):
        """
        Show a context menu for the cluster visibility list item at the given position.

        This method is called when the user right-clicks on an item in the cluster visibility list.
        It creates a context menu with options to change the color, rename, or save to Saved Masks.

        Args:
            position (QPoint): The position where the context menu should be displayed.
        """
        # Get the item at the position
        item = self.clusterVisibilityList.itemAt(position)
        if item is None:
            return

        # Create a context menu
        menu = QMenu()
        change_color_action = menu.addAction("Change Color")
        rename_action = menu.addAction("Rename")
        save_action = menu.addAction("Move to Saved Masks")

        # Show the menu and get the selected action
        action = menu.exec_(self.clusterVisibilityList.mapToGlobal(position))

        # Handle the selected action
        if action == change_color_action:
            self.change_cluster_color(item)
        elif action == rename_action:
            self.rename_cluster(item)
        elif action == save_action:
            self.save_live_cluster_to_saved(item)

    def change_cluster_color(self, item):
        """
        Change the color of a cluster.

        This method is called when the user selects the "Change Color" option from the context menu.
        It shows a color dialog and updates the cluster color in the _masks dictionary.

        Args:
            item (QListWidgetItem): The list item representing the cluster.
        """
        if self._masks is None or self.clusterview is None:
            return

        # Get the cluster ID from the item data
        cluster_id = int(item.data(Qt.UserRole))

        # Get the current color
        mask, current_color = self._masks.get(cluster_id, (None, None))
        if mask is None or current_color is None:
            return

        # Show a color dialog
        new_color = QColorDialog.getColor(current_color, self, f"Select Color for Cluster {cluster_id}")
        if not new_color.isValid():
            return

        # Update the color in the _masks dictionary
        self._masks[cluster_id] = (mask, new_color)

        # Update the item background color
        item.setBackground(new_color)

        # Set text color to black or white depending on the brightness of the background
        brightness = 0.299 * new_color.red() + 0.587 * new_color.green() + 0.114 * new_color.blue()
        text_color = Qt.black if brightness > 128 else Qt.white
        item.setForeground(text_color)

        # Clear the mask overlay cache to force recompute with the new color
        self.mask_overlays.clear()

        # Update the display
        self.show_label_image(self._clustered_image, self._num_labels)

    def rename_cluster(self, item):
        """
        Prompt the user to rename the selected cluster and update UI/mapping.
        """
        if item is None:
            return
        cluster_id = int(item.data(Qt.UserRole))
        current_name = self._cluster_names.get(cluster_id, item.text())
        new_name, ok = QInputDialog.getText(self, "Rename Cluster", f"Enter a new name for Cluster {cluster_id}:", text=str(current_name))
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Cluster name cannot be empty.")
            return
        # Save and update display text
        self._cluster_names[cluster_id] = new_name
        item.setText(new_name)

    def save_live_cluster_to_saved(self, item: QListWidgetItem):
        """Snapshot a live cluster mask into the Saved Masks list (session-only)."""
        if item is None or self._masks is None:
            return
        try:
            cluster_id = int(item.data(Qt.UserRole))
        except Exception:
            return
        if cluster_id not in self._masks:
            return
        mask, color = self._masks[cluster_id]
        # Deep copy the mask to prevent future mutation
        mask_copy = mask.copy()
        # Derive a name
        base_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        name = base_name
        # Ensure uniqueness across saved masks
        existing_names = {entry[2] for entry in self._saved_masks.values()}
        if name in existing_names:
            bump = 2
            while f"{name} [{bump}]" in existing_names:
                bump += 1
            name = f"{name} [{bump}]"
        # Assign new saved id
        saved_id = int(self._next_saved_id)
        self._next_saved_id += 1
        # Store
        self._saved_masks[saved_id] = (mask_copy, color, name)
        # Add to UI list
        saved_item = QListWidgetItem(name)
        saved_item.setData(Qt.UserRole, int(saved_id))
        saved_item.setFlags(saved_item.flags() | Qt.ItemIsUserCheckable)
        saved_item.setCheckState(Qt.Unchecked)
        try:
            saved_item.setBackground(color)
            brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            saved_item.setForeground(Qt.black if brightness > 128 else Qt.white)
        except Exception:
            pass
        self.savedMasksList.addItem(saved_item)


    def show_saved_context_menu(self, position):
        """Context menu for the Saved Masks list."""
        item = self.savedMasksList.itemAt(position)
        if item is None:
            return
        menu = QMenu()
        change_color_action = menu.addAction("Change Color")
        rename_action = menu.addAction("Rename")
        remove_action = menu.addAction("Remove from Saved")
        export_action = menu.addAction("Export")
        export_stats_action = menu.addAction("Export Intensity Stats (CSV)")
        action = menu.exec_(self.savedMasksList.mapToGlobal(position))
        if action == change_color_action:
            self.change_saved_mask_color(item)
        elif action == rename_action:
            self.rename_saved_mask(item)
        elif action == remove_action:
            self.remove_saved_mask(item)
        elif action == export_action:
            self.export_single_saved_mask(item)
        elif action == export_stats_action:
            try:
                sid = int(item.data(Qt.UserRole))
            except Exception:
                sid = None
            self.export_intensity_stats(saved_id=sid)

    def toggle_saved_visibility(self, item: QListWidgetItem):
        saved_id = int(item.data(Qt.UserRole)) if item is not None else None
        if saved_id is None:
            return
        if item.checkState() == Qt.Checked:
            self._saved_visible.add(saved_id)
        else:
            self._saved_visible.discard(saved_id)
        # Refresh display
        self.show_label_image(self._clustered_image, self._num_labels)

    def rename_saved_mask(self, item: QListWidgetItem):
        saved_id = int(item.data(Qt.UserRole))
        mask, color, current_name = self._saved_masks.get(saved_id, (None, None, None))
        if mask is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Saved Mask", "Enter a new name:", text=str(current_name))
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Saved mask name cannot be empty.")
            return
        # Enforce uniqueness across saved masks
        existing_names = {entry[2] for entry in self._saved_masks.values()} - {current_name}
        if new_name in existing_names:
            bump = 2
            base = new_name
            while f"{base} [{bump}]" in existing_names:
                bump += 1
            new_name = f"{base} [{bump}]"
        self._saved_masks[saved_id] = (mask, color, new_name)
        item.setText(new_name)

    def change_saved_mask_color(self, item: QListWidgetItem):
        saved_id = int(item.data(Qt.UserRole))
        mask, color, name = self._saved_masks.get(saved_id, (None, None, None))
        if mask is None:
            return
        new_color = QColorDialog.getColor(color, self, f"Select Color for {name}")
        if not new_color.isValid():
            return
        self._saved_masks[saved_id] = (mask, new_color, name)
        item.setBackground(new_color)
        brightness = 0.299 * new_color.red() + 0.587 * new_color.green() + 0.114 * new_color.blue()
        item.setForeground(Qt.black if brightness > 128 else Qt.white)
        # Clear saved overlay cache
        self.saved_mask_overlays.clear()
        self.show_label_image(self._clustered_image, self._num_labels)

    def remove_saved_mask(self, item: QListWidgetItem):
        saved_id = int(item.data(Qt.UserRole))
        # Remove overlay if present
        if saved_id in self.saved_overlay_items:
            try:
                if self.saved_overlay_items[saved_id].scene() == self.displayImage:
                    self.displayImage.removeItem(self.saved_overlay_items[saved_id])
            except Exception:
                pass
            self.saved_overlay_items.pop(saved_id, None)
        # Remove from data structures
        self._saved_masks.pop(saved_id, None)
        self._saved_visible.discard(saved_id)
        # Remove from UI
        row = self.savedMasksList.row(item)
        self.savedMasksList.takeItem(row)
        # Clear cache and refresh
        self.saved_mask_overlays.clear()
        self.show_label_image(self._clustered_image, self._num_labels)

    def export_single_saved_mask(self, item: QListWidgetItem):
        saved_id = int(item.data(Qt.UserRole))
        self._export_saved_mask_ids([saved_id])

    def export_saved_masks(self):
        if not self._saved_visible:
            QMessageBox.warning(self, "No Saved Masks Selected", "Please check one or more saved masks to export.")
            return
        self._export_saved_mask_ids(list(self._saved_visible))

    def export_intensity_stats(self, saved_id: int | None = None):
        """
        Export pixel intensity statistics under a selected Saved Mask for selected raw images as CSV.

        Columns: image_name, mask_id, mask_name, mean, median, min, max, range
        """
        # Ensure images loaded
        try:
            if self._image_set is None or self._image_set.num_images == 0:
                QMessageBox.warning(self, "No Images Loaded", "Please load raw images before exporting statistics.")
                return
        except Exception:
            QMessageBox.warning(self, "Image Set Error", "The image set is not properly initialized. Please load images first.")
            return

        # Ensure there is at least one saved mask
        if not self._saved_masks:
            QMessageBox.warning(self, "No Saved Masks", "There are no saved masks available. Save a mask first.")
            return

        # Determine which saved mask to use
        sid = None
        if isinstance(saved_id, int):
            sid = int(saved_id)
        else:
            # Try current selection in the Saved Masks list
            current_item = self.savedMasksList.currentItem() if hasattr(self, 'savedMasksList') else None
            sid = int(current_item.data(Qt.UserRole)) if current_item is not None else None
        if sid is None or sid not in self._saved_masks:
            # Prompt user to pick one from names
            items = []
            id_for_row = []
            for k, (_mask, _color, nm) in self._saved_masks.items():
                items.append(f"{k}: {nm}")
                id_for_row.append(int(k))
            choice, ok = QInputDialog.getItem(self, "Select Saved Mask", "Saved Mask:", items, 0, False)
            if not ok:
                return
            # Parse id from the prefix before ':'
            try:
                sid = int(choice.split(":", 1)[0].strip())
            except Exception:
                QMessageBox.warning(self, "Selection Error", "Could not determine selected saved mask.")
                return

        entry = self._saved_masks.get(int(sid))
        if not entry:
            QMessageBox.warning(self, "Mask Error", "Selected saved mask not found.")
            return
        roi_mask, _color, mask_name = entry

        # Validate mask shape against images
        try:
            img_h, img_w = self._image_set.image_shape
        except Exception:
            QMessageBox.warning(self, "Image Set Error", "Image shapes are unavailable. Reload images and try again.")
            return
        if roi_mask.shape != (img_h, img_w):
            QMessageBox.warning(self, "Shape Mismatch", f"Saved mask shape {roi_mask.shape} does not match image shape {(img_h, img_w)}.")
            return

        # Prompt to select which images to include
        select_dlg = ImageSelectDlg(self.fileNameList, self._image_set, checked=True)
        select_dlg.setModal(True)
        select_dlg.exec_()
        selected_mask = getattr(select_dlg, 'selected_mask', None)
        if selected_mask is None or not np.any(selected_mask):
            QMessageBox.information(self, "No Images Selected", "No images were selected for export.")
            return

        # Prompt for CSV save path
        csv_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);")
        if not csv_path:
            return
        if not csv_path.lower().endswith('.csv'):
            csv_path += '.csv'

        # Compute statistics and write CSV
        images = self._image_set.images  # shape (N,H,W), float
        header = ["image_name", "mask_id", "mask_name", "mean", "median", "min", "max", "range"]
        empty_mask = not bool(np.any(roi_mask))
        try:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for idx, use in enumerate(selected_mask):
                    if not bool(use):
                        continue
                    img = images[idx]
                    values = img[roi_mask]
                    if values.size == 0 or empty_mask:
                        mean = median = vmin = vmax = vrange = float('nan')
                    else:
                        # Ensure finite values
                        vals = values[np.isfinite(values)]
                        if vals.size == 0:
                            mean = median = vmin = vmax = vrange = float('nan')
                        else:
                            vmin = float(np.min(vals))
                            vmax = float(np.max(vals))
                            mean = float(np.mean(vals))
                            median = float(np.median(vals))
                            vrange = float(vmax - vmin)
                    writer.writerow([self.fileNameList[idx], int(sid), str(mask_name), mean, median, vmin, vmax, vrange])
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write CSV: {e}")
            return

        QMessageBox.information(self, "Success", f"Intensity statistics exported to {csv_path}")

    def _export_saved_mask_ids(self, saved_ids: list[int]):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Masks")
        if not output_dir:
            return
        file_format = self.exportFormatCombo.currentText().lower()
        progress = QProgressDialog("Exporting saved masks...", "Cancel", 0, len(saved_ids), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)
        for i, sid in enumerate(saved_ids):
            if progress.wasCanceled():
                break
            entry = self._saved_masks.get(int(sid))
            if not entry:
                progress.setValue(i + 1)
                continue
            mask, _color, name = entry
            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or f"saved_{sid}"
            output_path = os.path.join(output_dir, f"{safe_name}.{file_format}")
            try:
                mask_array = mask.astype(np.uint8) * 255
                img = Image.fromarray(mask_array, mode='L')
                img.save(output_path)
            except Exception as e:
                print(f"Failed to export saved mask {sid}: {e}")
            progress.setValue(i + 1)
        if not progress.wasCanceled():
            QMessageBox.information(self, "Success", f"Saved masks exported to {output_dir}")

    def toggle_cluster_visibility(self, item):
        cluster_id = int(item.data(Qt.UserRole))
        print(f"Toggling visibility for cluster {cluster_id}, check state: {item.checkState()}")
        if item.checkState() == Qt.Checked:
            self._visible_clusters.add(cluster_id)
            print(f"Showing mask for cluster {cluster_id}")
        else:
            self._visible_clusters.discard(cluster_id)
            print(f"Hiding mask for cluster {cluster_id}")
        self.show_label_image(self._clustered_image, self._num_labels)

    def merge_selected_clusters(self):
        """
        Merge selected clusters into a single cluster.

        This method gathers the checked clusters in the Cluster Mask Visibility list,
        calls the merge_clusters method in Cluster.py with those IDs, and invokes
        the on_cluster_callback with the updated labels/settings.
        """
        if self.clusterview is None or self._masks is None:
            QMessageBox.warning(self, "No Clustering Data", "Please run initial clustering first.")
            return

        # Gather checked cluster ids from the visibility list
        checked_ids = []
        for i in range(self.clusterVisibilityList.count()):
            item = self.clusterVisibilityList.item(i)
            if item.checkState() == Qt.Checked:
                checked_ids.append(int(item.data(Qt.UserRole)))

        if len(checked_ids) < 2:
            QMessageBox.warning(self, "Insufficient Clusters Selected", "Please check at least two clusters to merge in the Cluster Mask Visibility list.")
            return

        # Call the merge_clusters method in Cluster.py
        new_labels, new_settings = self.clusterview.merge_clusters(checked_ids)

        if new_labels is None:
            QMessageBox.warning(self, "Merge Failed", "Failed to merge the selected clusters. Please ensure all selected clusters are valid.")
            return

        # Call the callback with the new labels and settings
        self.on_cluster_callback(new_labels, new_settings)

        # Show success message
        QMessageBox.information(self, "Merge Successful", f"Successfully merged {len(checked_ids)} clusters into a single cluster.")

    def run_iterative_clustering(self):
        # Iterative clustering UI has been removed per user request; method retained for compatibility
        QMessageBox.information(self, "Iterative Clustering Disabled", "Iterative clustering controls have been removed. Use cluster masks as ROI externally or via future workflows.")
        return

    def undo_clustering(self):
        if self.clusterview and self.clusterview.undo_clustering():
            self.on_cluster_callback(self.clusterview.labels, KMeansSettings(n_clusters=len(np.unique(self.clusterview.labels)), init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42))
        else:
            QMessageBox.warning(self, "No Undo", "No previous clustering state available to undo.")

    def update_mask_opacity(self, value):
        self._mask_opacity = value
        # Clear mask overlay cache to force recompute with new opacity
        self.mask_overlays.clear()
        self.saved_mask_overlays.clear()
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

            # Build output path using custom name if available
            custom_name = self._cluster_names.get(int(cluster_id))
            if custom_name:
                safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', custom_name).strip('_') or f"cluster_{cluster_id}"
                output_path = os.path.join(output_dir, f"{safe_name}")
            else:
                output_path = os.path.join(output_dir, f"cluster_{cluster_id}_mask")

            # Use Cluster class to export the mask
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

        # Remove overlays for deselected clusters (live)
        clusters_to_remove = set(self.overlay_items.keys()) - self._visible_clusters
        for cluster_id in clusters_to_remove:
            if cluster_id in self.overlay_items:
                item = self.overlay_items[cluster_id]
                try:
                    if item.scene() == self.displayImage:
                        self.displayImage.removeItem(item)
                except Exception:
                    pass
                finally:
                    del self.overlay_items[cluster_id]

        # Remove overlays for deselected saved masks
        saved_to_remove = set(self.saved_overlay_items.keys()) - self._saved_visible
        for sid in saved_to_remove:
            if sid in self.saved_overlay_items:
                item = self.saved_overlay_items[sid]
                try:
                    if item.scene() == self.displayImage:
                        self.displayImage.removeItem(item)
                except Exception:
                    pass
                finally:
                    del self.saved_overlay_items[sid]

        # Handle mask overlays for visible live clusters
        if self._masks and self._visible_clusters:
            first_mask, _ = next(iter(self._masks.values()))
            height, width = first_mask.shape

            # Calculate optimal scale factor
            if self.clusterview is not None:
                scale_factor = self.clusterview.calculate_optimal_scale_factor(height, width)
            else:
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
                    if new_width != width or new_height != height:
                        mask_small = cv2.resize(mask.astype(np.uint8), (new_width, new_height),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask_small = mask

                    overlay = QImage(new_width, new_height, QImage.Format_ARGB32)
                    overlay.fill(Qt.transparent)

                    mask_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                    mask_data[mask_small, 0] = color.blue()
                    mask_data[mask_small, 1] = color.green()
                    mask_data[mask_small, 2] = color.red()
                    mask_data[mask_small, 3] = self._mask_opacity

                    overlay_data = mask_data.tobytes()
                    overlay = QImage(overlay_data, new_width, new_height, QImage.Format_ARGB32)

                overlay_pixmap = QPixmap.fromImage(overlay).scaled(2000, 5000, Qt.KeepAspectRatio)
                self.mask_overlays[cache_key] = overlay_pixmap

                if cluster_id in self.overlay_items:
                    try:
                        if self.overlay_items[cluster_id].scene() == self.displayImage:
                            self.displayImage.removeItem(self.overlay_items[cluster_id])
                    except Exception:
                        pass

                item = self.displayImage.addPixmap(overlay_pixmap)
                self.overlay_items[cluster_id] = item

        # Now render saved masks above live masks
        if self._saved_masks and self._saved_visible:
            # infer shape from any saved mask
            any_entry = next(iter(self._saved_masks.values()))
            smask = any_entry[0]
            height, width = smask.shape
            if self.clusterview is not None:
                scale_factor = self.clusterview.calculate_optimal_scale_factor(height, width)
            else:
                max_pixels = 500000
                scale_factor = np.sqrt(max_pixels / (height * width)) if height * width > max_pixels else 1.0
            if scale_factor < 1.0:
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
            else:
                new_height, new_width = height, width

            for sid in list(self._saved_visible):
                entry = self._saved_masks.get(int(sid))
                if not entry:
                    continue
                mask, color, _name = entry
                cache_key = (int(sid), self._mask_opacity, new_width, new_height)
                if cache_key in self.saved_mask_overlays:
                    overlay_pixmap = self.saved_mask_overlays[cache_key]
                    if sid not in self.saved_overlay_items:
                        item = self.displayImage.addPixmap(overlay_pixmap)
                        self.saved_overlay_items[sid] = item
                    continue
                # Create overlay
                if self.clusterview is not None:
                    overlay = self.clusterview.create_mask_overlay(
                        mask, color, self._mask_opacity,
                        target_width=new_width, target_height=new_height
                    )
                else:
                    if new_width != width or new_height != height:
                        mask_small = cv2.resize(mask.astype(np.uint8), (new_width, new_height),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask_small = mask
                    overlay = QImage(new_width, new_height, QImage.Format_ARGB32)
                    overlay.fill(Qt.transparent)
                    mask_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                    mask_data[mask_small, 0] = color.blue()
                    mask_data[mask_small, 1] = color.green()
                    mask_data[mask_small, 2] = color.red()
                    mask_data[mask_small, 3] = self._mask_opacity
                    overlay_data = mask_data.tobytes()
                    overlay = QImage(overlay_data, new_width, new_height, QImage.Format_ARGB32)
                overlay_pixmap = QPixmap.fromImage(overlay).scaled(2000, 5000, Qt.KeepAspectRatio)
                self.saved_mask_overlays[cache_key] = overlay_pixmap
                if sid in self.saved_overlay_items:
                    try:
                        if self.saved_overlay_items[sid].scene() == self.displayImage:
                            self.displayImage.removeItem(self.saved_overlay_items[sid])
                    except Exception:
                        pass
                item = self.displayImage.addPixmap(overlay_pixmap)
                self.saved_overlay_items[sid] = item

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
