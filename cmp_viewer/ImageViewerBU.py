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

__version__ = '1.0'
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
    _visible_clusters = set()  # Track which clusters' masks are visible (now for reference)
    _mask_opacity = 100  # Default opacity (0-255)

    """View Gui"""

    def __init__(self, starting_images_folder=None):
        """View Initializer"""
        super().__init__()

        self._image_set = models.ImageSet()
        # Some main properties of the Window
        self.clusterview = None
        self.setWindowTitle('ImageViewer')

        # Set central widget and the general layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.generalLayout = QHBoxLayout()
        self.centralWidget.setLayout(self.generalLayout)

        self.centralWidget.setMinimumSize(1500, 1000)

        """Don't forget to actually create a display"""
        self._createDisplay()
        self._createMenuBar()
        self._createViewList()
        self._createIterativeClusteringControls()

        if starting_images_folder is not None:
            if not os.path.isdir(starting_images_folder):
                print(f"Starting images folder is not a directory: {starting_images_folder}")
            else:
                search_path = os.path.join(starting_images_folder, "*.tif")
                filenames = glob.glob(search_path)
                self.open_images(filenames)

    def _createMenuBar(self):
        """Create a menubar"""
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)
        fileMenu = menuBar.addMenu('File')
        clusterMenu = menuBar.addMenu('Cluster')

        """create menu items"""
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
        closeAct.triggered.connect(sys.exit)
        fileMenu.addAction(closeAct)

        """Clustering options"""
        selectImagesAct = QAction('Select Images', self)
        selectImagesAct.triggered.connect(self.selectClustImages)
        clusterMenu.addAction(selectImagesAct)

    def _createViewList(self):
        # create file view list
        self.ViewList_Box = QtWidgets.QGroupBox('Open Images')
        self.ViewList_Box.setMinimumSize(400, 200)
        self.ViewList_Layout = QVBoxLayout()
        self.ViewList_Box.setLayout(self.ViewList_Layout)

        self.ViewList_Layout.setSpacing(2)
        self.ViewList_Layout.setContentsMargins(2, 2, 2, 2)
        self.generalLayout.addWidget(self.ViewList_Box)

    def _createDisplay(self):
        # Create display widget
        self.display = QScrollArea()
        self.displayView = QGraphicsView()
        self.displayImage = QGraphicsScene()

        # Set up display window properties
        self.display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setWidgetResizable(True)

        self.displayView.setScene(self.displayImage)  # Set the scene of the QGraphicsView
        self.display.setWidget(self.displayView)  # Set the QGraphicsView as the widget of the QScrollArea
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(1100, 1000)

        # add this display to the general layout
        self.generalLayout.addWidget(self.display)

    def _createIterativeClusteringControls(self):
        # Create a group box for iterative clustering controls
        self.iterativeClusterBox = QtWidgets.QGroupBox('Iterative Clustering')
        self.iterativeClusterBox.setMinimumSize(400, 350)  # Increased size for new controls
        self.iterativeClusterLayout = QVBoxLayout()
        self.iterativeClusterBox.setLayout(self.iterativeClusterLayout)

        # Combo box to select a cluster mask
        self.clusterSelectCombo = QComboBox()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.iterativeClusterLayout.addWidget(self.clusterSelectCombo)

        # Input for number of clusters
        self.subClustersInput = QLineEdit()
        self.subClustersInput.setPlaceholderText("Number of sub-clusters")
        self.iterativeClusterLayout.addWidget(self.subClustersInput)

        # Button to run iterative clustering
        self.iterativeClusterButton = QPushButton("Run Iterative Clustering")
        self.iterativeClusterButton.clicked.connect(self.run_iterative_clustering)
        self.iterativeClusterLayout.addWidget(self.iterativeClusterButton)

        # List widget for cluster mask visibility (now for reference or future use)
        self.clusterVisibilityList = QListWidget()
        self.clusterVisibilityList.setMinimumHeight(100)
        self.iterativeClusterLayout.addWidget(QLabel("Cluster Mask Visibility (Reference)"))
        self.iterativeClusterLayout.addWidget(self.clusterVisibilityList)

        # Slider to control mask opacity (optional, can be removed if not needed)
        self.opacitySlider = QSlider(Qt.Horizontal)
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(255)
        self.opacitySlider.setValue(self._mask_opacity)
        self.opacitySlider.setTickPosition(QSlider.TicksBelow)
        self.opacitySlider.setTickInterval(25)
        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)
        self.iterativeClusterLayout.addWidget(QLabel("Mask Opacity"))
        self.iterativeClusterLayout.addWidget(self.opacitySlider)

        # Combo box for export file format
        self.exportFormatCombo = QComboBox()
        self.exportFormatCombo.addItems(["PNG", "BMP", "TIFF"])
        self.exportFormatCombo.setCurrentText("PNG")
        self.iterativeClusterLayout.addWidget(QLabel("Export Format"))
        self.iterativeClusterLayout.addWidget(self.exportFormatCombo)

        # Button to export cluster masks
        self.exportMasksButton = QPushButton("Export Cluster Masks")
        self.exportMasksButton.clicked.connect(self.export_cluster_masks)
        self.iterativeClusterLayout.addWidget(self.exportMasksButton)

        # Button for undo
        self.undoButton = QPushButton("Undo Clustering")
        self.undoButton.clicked.connect(self.undo_clustering)
        self.iterativeClusterLayout.addWidget(self.undoButton)

        self.generalLayout.addWidget(self.iterativeClusterBox)

    def chooseGrayscaleImage(self, index):
        img_array = self.rawImages[index]

        gray1D = img_array.tobytes()
        qImg = QImage(gray1D, img_array.shape[1], img_array.shape[0], QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qImg)
        self.displayImage.clear()  # Clear previous image
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseRedImage(self, index):
        self.r_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.clear()  # Clear previous image
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseGreenImage(self, index):
        self.g_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.clear()  # Clear previous image
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseBlueImage(self, index):
        self.b_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.clear()  # Clear previous image
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def on_open_images_menu(self):
        results = QFileDialog.getOpenFileNames(self, self.tr("Select image(s) to open"))
        self.open_images(results[0])

    def open_images(self, filenames: list[str]):
        self.ImportLayout = QVBoxLayout()
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()
        self.ViewList_Layout.addLayout(self.ImportLayout)

        for index, filename in enumerate(filenames):
            basefileName = os.path.basename(filename)
            simpleName = os.path.splitext(basefileName)[0]
            self.fileNameList.append(simpleName)
            self.importImageWrapper(filename)
            self.colorRBs(filename, index)

        self.chooseGrayscaleImage(0)

    def importImageWrapper(self, fileName):
        '''
        Imports images into UI
        :param str fileName: Filename designated by openImages
        :return: viewable image with color select radio buttons
        :rtype: numpy nd array
        '''
        image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        self.rawImages.append(image)

        image = nornir_imageregistration.LoadImage(fileName)
        image_float = image.astype(np.float32)
        self._image_set.add_image(image_float)

    def colorRBs(self, fileName, index):
        row = QtWidgets.QGroupBox()
        rowLayout = QHBoxLayout()

        # Add Filenames associated with RBs
        basefileName = os.path.basename(fileName)
        simpleName = os.path.splitext(basefileName)[0]
        rowLayout.addWidget(QLabel(simpleName))

        # Adding buttons for grayscale
        grayRadioButton = QRadioButton('gray')
        grayRadioButton.toggled.connect(lambda: self.chooseGrayscaleImage(index))
        rowLayout.addWidget(grayRadioButton)
        self.column1.addButton(grayRadioButton)

        # Adding buttons for red
        redRadioButton = QRadioButton("R")
        redRadioButton.toggled.connect(lambda: self.chooseRedImage(index))
        rowLayout.addWidget(redRadioButton)
        self.redRBlist.addButton(redRadioButton)

        # Adding buttons for green
        greenRadioButton = QRadioButton("G")
        greenRadioButton.toggled.connect(lambda: self.chooseGreenImage(index))
        rowLayout.addWidget(greenRadioButton)
        self.greenRBlist.addButton(greenRadioButton)
        # Adding buttons for blue
        blueRadioButton = QRadioButton("B")
        blueRadioButton.toggled.connect(lambda: self.chooseBlueImage(index))
        rowLayout.addWidget(blueRadioButton)
        self.blueRBlist.addButton(blueRadioButton)

        row.setLayout(rowLayout)
        self.ImportLayout.addWidget(row)

    def save_clustered_image(self):
        """Save the clustered image to a file"""
        if self._clustered_image is None:
            QtWidgets.QMessageBox.warning(self, "No Clustered Image", "No clustered image available to save.")
            return

        # Open a file dialog to select save location, default to BMP
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Clustered Image",
            "",
            "BMP Files (*.bmp);;PNG Files (*.png);;All Files (*)"
        )

        if file_name:
            # Ensure the file has a .bmp or .png extension if none is provided
            if not file_name.lower().endswith(('.bmp', '.png')):
                file_name += '.bmp'
            # Save the clustered image
            self._clustered_image.save(file_name)
            QtWidgets.QMessageBox.information(self, "Success", f"Clustered image saved to {file_name}")

    def close_images(self):
        """Close all opened images and reset the UI"""
        # Disconnect the opacity slider to prevent unwanted updates
        try:
            self.opacitySlider.valueChanged.disconnect()
        except Exception:
            pass  # Signal might not be connected

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
        self._mask_opacity = 100  # Reset to default
        if self.clusterview:
            self.clusterview.undo_stack.clear()
        self._image_set = models.ImageSet()  # Reinitialize ImageSet
        self.displayImage.clear()  # Clear the display

        # Clear iterative clustering controls
        self.clusterSelectCombo.clear()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.subClustersInput.clear()
        self.clusterVisibilityList.clear()
        self.opacitySlider.setValue(self._mask_opacity)
        self.exportFormatCombo.setCurrentText("PNG")

        # Remove all widgets from ImportLayout
        while self.ImportLayout.count():
            item = self.ImportLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.ImportLayout = QVBoxLayout()  # Reinitialize ImportLayout
        self.ViewList_Layout.addLayout(self.ImportLayout)
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()

        # Reconnect the opacity slider
        self.opacitySlider.valueChanged.connect(self.update_mask_opacity)
        QMessageBox.information(self, "Success", "All images have been closed.")

    def reset_viewer(self):
        """Reset the entire viewer to its initial state"""
        # Disconnect the opacity slider to prevent unwanted updates
        try:
            self.opacitySlider.valueChanged.disconnect()
        except Exception:
            pass  # Signal might not be connected

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
        self._mask_opacity = 100  # Reset to default
        if self.clusterview:
            self.clusterview.undo_stack.clear()
        self._image_set = models.ImageSet()  # Reinitialize ImageSet
        self.displayImage.clear()  # Clear the display

        # Clear iterative clustering controls
        self.clusterSelectCombo.clear()
        self.clusterSelectCombo.addItem("Select Cluster")
        self.subClustersInput.clear()
        self.clusterVisibilityList.clear()
        self.opacitySlider.setValue(self._mask_opacity)
        self.exportFormatCombo.setCurrentText("PNG")

        # Remove all widgets from generalLayout and recreate
        while self.generalLayout.count():
            item = self.generalLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._createDisplay()
        self._createMenuBar()
        self._createViewList()
        self._createIterativeClusteringControls()
        QMessageBox.information(self, "Success", "Viewer has been reset.")

    def selectClustImages(self):
        '''
        Select images for clustering using GUI
        '''
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
        pillow_img = cmp_viewer.numpy_labels_to_pillow_image(labels)
        if isinstance(settings, KMeansSettings):
            # Store masks from the Cluster instance
            self._masks = self.clusterview.masks
            # Store the clustered image and number of labels
            self._clustered_image = pillow_img
            self._num_labels = len(np.unique(labels))

            # Add masks to rawImages and update UI
            self.rawImages.extend([(mask * 255).astype(np.uint8) for mask, _ in self._masks.values()])
            mask_indices = list(range(len(self.rawImages) - len(self._masks), len(self.rawImages)))
            for i, cluster_id in enumerate(np.unique(labels)):
                if i < len(mask_indices):
                    simpleName = f"Cluster_{cluster_id}_Mask"
                    self.fileNameList.append(simpleName)
                    self.colorRBs(f"{simpleName}.tif", mask_indices[i])

            # Update the cluster selection combo box
            self.clusterSelectCombo.clear()
            self.clusterSelectCombo.addItem("Select Cluster")
            for cluster_id in np.unique(labels):
                self.clusterSelectCombo.addItem(f"Cluster {cluster_id}")

            # Update the visibility list (now for reference)
            self.update_cluster_visibility_list(labels)
            self.show_label_image(pillow_img, len(np.unique(labels)))

    def update_cluster_visibility_list(self, labels: NDArray[int]):
        """Update the list of cluster visibility toggles (now for reference)"""
        self.clusterVisibilityList.clear()
        for cluster_id in np.unique(labels):
            item = QListWidgetItem(f"Cluster {cluster_id}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.clusterVisibilityList.addItem(item)
        # No longer connect to toggle_cluster_visibility since masks are handled as images

    def run_iterative_clustering(self):
        """Run k-means clustering on the selected mask"""
        if self.clusterview is None or self._masks is None:
            QMessageBox.warning(self, "No Clustering Data", "Please run initial clustering first.")
            return

        # Get selected cluster
        selected_cluster = self.clusterSelectCombo.currentText()
        if selected_cluster == "Select Cluster":
            QMessageBox.warning(self, "Invalid Selection", "Please select a cluster to refine.")
            return

        cluster_id = int(selected_cluster.split()[-1])
        mask, _ = self._masks.get(cluster_id, (None, None))
        if mask is None:
            QMessageBox.warning(self, "Invalid Mask", "Selected cluster mask is not available.")
            return

        # Get number of sub-clusters
        try:
            n_sub_clusters = int(self.subClustersInput.text())
            if n_sub_clusters < 2:
                raise ValueError("Number of sub-clusters must be at least 2.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of sub-clusters (at least 2).")
            return

        # Run iterative clustering
        new_labels, new_settings = self.clusterview.cluster_on_mask(mask, n_sub_clusters)
        if new_labels is None:
            QMessageBox.warning(self, "No Pixels", "No pixels in the selected mask region for clustering.")
            return

        # Update with new clustering results
        self.on_cluster_callback(new_labels, new_settings)

    def undo_clustering(self):
        """Trigger undo in the Cluster instance"""
        if self.clusterview and self.clusterview.undo_clustering():
            self.on_cluster_callback(self.clusterview.labels, KMeansSettings(n_clusters=len(np.unique(self.clusterview.labels)), init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42))
        else:
            QMessageBox.warning(self, "No Undo", "No previous clustering state available to undo.")

    def update_mask_opacity(self, value):
        """Update the mask overlay opacity based on the slider value (optional, can be removed)"""
        self._mask_opacity = value
        if self._clustered_image is not None and self._num_labels is not None:
            self.show_label_image(self._clustered_image, self._num_labels)
        else:
            print("Cannot update mask opacity: Clustered image or number of labels is not set.")

    def export_cluster_masks(self):
        """Export selected cluster masks as images with chosen format"""
        if self._masks is None:
            QMessageBox.warning(self, "No Masks Available", "Please run clustering to generate masks before exporting.")
            return

        if not self._visible_clusters:
            QMessageBox.warning(self, "No Clusters Selected", "Please select at least one cluster for export by checking the visibility list.")
            return

        # Prompt user to select a directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Cluster Masks")
        if not output_dir:
            return  # User canceled the dialog

        # Get the selected file format
        file_format = self.exportFormatCombo.currentText().lower()
        if file_format == "tiff":
            file_ext = ".tif"
        else:
            file_ext = f".{file_format}"

        # Set up progress dialog
        progress = QProgressDialog("Exporting cluster masks...", "Cancel", 0, len(self._visible_clusters), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(1000)  # Show after 1 second

        # Export each selected mask
        for i, cluster_id in enumerate(self._visible_clusters):
            if progress.wasCanceled():
                break
            mask, _ = self._masks.get(cluster_id, (None, None))
            if mask is None:
                continue

            # Convert boolean mask to grayscale (True -> 255, False -> 0)
            mask_array = (mask * 255).astype(np.uint8)
            # Convert to PIL Image
            mask_image = Image.fromarray(mask_array, mode='L')
            # Save the mask
            output_path = os.path.join(output_dir, f"cluster_{cluster_id}_mask{file_ext}")
            mask_image.save(output_path)
            progress.setValue(i + 1)

        if not progress.wasCanceled():
            QMessageBox.information(self, "Success", f"Cluster masks exported to {output_dir}")
        else:
            QMessageBox.information(self, "Export Canceled", "Export process was canceled.")

    def show_label_image(self, img, num_labels: int):
        # Guard against None values
        if img is None or num_labels is None:
            print("Cannot show label image: Image or num_labels is None.")
            return

        # Store the clustered image and number of labels
        self._clustered_image = img
        self._num_labels = num_labels

        # Create the color table (same as used for display)
        self._color_table = [qRgb(int((i/num_labels) * 255), int((i/num_labels) * 255), int((i/num_labels-1) * 255)) for i in range(num_labels)]

        # Convert the color table to a flat list of RGB values for PIL palette
        palette = []
        for rgb in self._color_table:
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            palette.extend([r, g, b])

        # Ensure the image is in mode 'P' and apply the palette
        if self._clustered_image.mode != 'P':
            self._clustered_image = self._clustered_image.convert('P')
        self._clustered_image.putpalette(palette)

        # Convert PIL Image to QImage for display
        qimage = QImage(self._clustered_image.tobytes(), self._clustered_image.size[0], self._clustered_image.size[1], QImage.Format_Indexed8)
        qimage.setColorCount(num_labels)
        qimage.setColorTable(self._color_table)

        # Convert QImage to QPixmap for the base image
        pixmap = QPixmap.fromImage(qimage)
        self.displayImage.clear()  # Clear previous image
        self.displayImage.addPixmap(pixmap.scaled(2000, 5000, Qt.KeepAspectRatio))

# Client code
def main():
    """Main function"""
    starting_images_folder = os.environ['DEBUG_IMAGES_FOLDER'] if 'DEBUG_IMAGES_FOLDER' in os.environ else None
    # Create QApplication instance (remember 1 per project)
    CMPViewer = QApplication(sys.argv)
    # show the UI
    view = ImageViewerUi(starting_images_folder)
    view.show()
    # Execute the main loop
    sys.exit(CMPViewer.exec_())

if __name__ == '__main__':
    main()