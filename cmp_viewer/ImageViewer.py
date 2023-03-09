# Filename: ImageViewer.py
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
import sys
import os
import glob
from cmp_viewer.rgb import *
from cmp_viewer.clusterImgSelect import *
from cmp_viewer.Cluster import *
import nornir_imageregistration
from cmp_viewer import models
from PIL import Image
import typing

from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
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
from functools import partial

__version__ = '1.0'
__author__ = "RL Pfeiffer & NQN Studios"


class ImageViewerUi(QMainWindow):
    rawImages = []
    fileNameList = []
    r_image = None
    g_image = None
    b_image = None

    """View Gui"""

    def __init__(self, starting_images_folder=None):
        """View Initializer"""
        super().__init__()

        self._image_set = models.ImageSet()
        # Some main properties of the Window
        self.clusterview = None
        self.setWindowTitle('ImageViewer')

        # Set central widget and the general layout
        # Create a QWidget object to act as the container for your QHBoxLayout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        # Create your QHBoxLayout and add it to the central widget
        self.generalLayout = QHBoxLayout()
        self.centralWidget.setLayout(self.generalLayout)

        # Set the minimum size of the central widget
        self.centralWidget.setMinimumSize(1500, 1000)

        """Don't forget to actually create a display"""
        self._createDisplay()
        self._createMenuBar()
        self._createViewList()

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
        self.menuBar = QMenuBar()
        menuBar.setNativeMenuBar(False)
        fileMenu = menuBar.addMenu('File')
        clusterMenu = menuBar.addMenu('Cluster')
        self.generalLayout.addWidget(self.menuBar)

        """create menu items"""
        openAct = QAction('Open Images', self)
        openAct.setShortcut('Ctrl+O')
        openAct.triggered.connect(self.on_open_images_menu)
        fileMenu.addAction(openAct)

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
        self.ViewList_Layout = QtWidgets.QVBoxLayout()
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

    def chooseGrayscaleImage(self, index):
        img_array = self.rawImages[index]

        gray1D = img_array.tobytes()
        qImg = QImage(gray1D, img_array.shape[1], img_array.shape[0], QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qImg)
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseRedImage(self, index):
        self.r_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseGreenImage(self, index):
        self.g_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.addPixmap((pixmap).scaled(2000, 5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseBlueImage(self, index):
        self.b_image = self.rawImages[index]
        composite = create_composite_image(self.rawImages, self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
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

        image_float = nornir_imageregistration.LoadImage(fileName, dtype=np.float32)
        self._image_set.add_image(image_float)

    def colorRBs(self, fileName, index):
        row = QtWidgets.QGroupBox()
        rowLayout = QtWidgets.QHBoxLayout()

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

    # Select images for clustering using GUI
    def selectClustImages(self):
        '''
        Select images for clustering using GUI
        '''
        select_dlg = ImageSelectDlg(self.fileNameList, self._image_set)
        select_dlg.setModal(True)
        select_dlg.exec_()

        self.clusterview = Cluster(self.fileNameList, self._image_set, select_dlg.selected_mask, self.on_cluster_callback)
        self.clusterview.show()

    def on_cluster_callback(self, labels: NDArray[int], settings: typing.Any):

        pillow_img = cmp_viewer.numpy_labels_to_pillow_image(labels)
        if isinstance(settings, KMeansSettings):
            self.show_label_image(pillow_img, settings.n_clusters)


    def show_label_image(self, img, num_labels: int):

        # Convert PIL Image to QImage
        qimage = QImage(img.tobytes(), img.size[0], img.size[1], QImage.Format_Indexed8)
        qimage.setColorCount(num_labels)
        qimage.setColorTable([qRgb(int((i/num_labels) * 255), int((i/num_labels) * 255), int((i/num_labels-1) * 255)) for i in range(num_labels)])

        # Convert QImage to QPixmap and create QGraphicsPixmapItem
        pixmap = QPixmap.fromImage(qimage)
        ClusterMask = QGraphicsPixmapItem(pixmap)

        # Add QGraphicsPixmapItem to parent QGraphicsScene

        #self.parent.displayImage.addPixmap(ClusterMask.scaled(2000, 5000, Qt.KeepAspectRatio))
        self.displayImage.addPixmap(ClusterMask.pixmap().scaled(2000, 5000, Qt.KeepAspectRatio))

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