# Filename: ImageViewer.py
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
from genericpath import isfile
import sys
import os

from numpy import concatenate
from rgb import *
from cluster import *
import collections.abc
import nornir_imageregistration
import imageset

from PyQt5.QtWidgets import QApplication
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

DEBUG = True #if 'DEBUG' in sys.environment  

class ImageViewerUi(QMainWindow):
    rawImages = []
    fileNameList = []
    r_image = None
    g_image = None
    b_image = None
 

    """View Gui"""
    def __init__(self):
        """View Initializer"""
        super().__init__()
        self.npImages = None
        #Some main properties of the Window
        self.setWindowTitle('ImageViewer')

        #Set central widget and the general layout
        self.generalLayout = QHBoxLayout()
        self._centralWidget = QScrollArea()
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        self._centralWidget.setMinimumSize(1500, 800)
 
        """Don't forget to actually create a display"""
        self._createDisplay()
        self._createMenuBar()
        self._createViewList()

        self.ImportLayout = QVBoxLayout()
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()
        self.ViewList_Layout.addLayout(self.ImportLayout)

        self._model = None

        if DEBUG:
            (imageStack, imageNames) = self.LoadFilesOrDirectories('/Users/rpfeiffer/CodingProjects/CMPViewer/test_images') 
            self.setmodel(imageset.ImageSet(imageStack), imageNames) 

    @property
    def model(self):
        return self._model
 
    def setmodel(self, value, names):
        #Delete any UI if it exists
        self._model = value
        if self._model is None:
            return 
        self.rawImages = []
        for i in range(0, value.numImages):
            qtImage = array2qimage(value.images[i,:,:])
            self.rawImages.append(qtImage)
            row = self.createImageListEntry(names[i],i)
            self.ImportLayout.addWidget(row)

        #Build the image list and any other UI for the model
        #self.fileNameList.append(simpleName) 
        #self.imageListLayout(fileName, len(self.fileNameList) -1 )



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
        openAct.triggered.connect(self.openImages)
        fileMenu.addAction(openAct)

        closeAct = QAction('Close', self)
        closeAct.setShortcut('Ctrl+Q')
        closeAct.triggered.connect(sys.exit)
        fileMenu.addAction(closeAct)

        """Clustering options"""
        #selectImagesAct = QAction('Select Images', self)
        #selectImagesAct.triggered.connect(self.selectClustImages)
        #clusterMenu.addAction(selectImagesAct)
        
        clusterKMeansAct = QAction('K-means', self)
        clusterKMeansAct.triggered.connect(self.clusterKMeans)
        clusterMenu.addAction(clusterKMeansAct)
 
    def _createViewList(self):
        #create file view list
        self.ViewList_Box = QtWidgets.QGroupBox('Open Images')
        self.ViewList_Box.setMinimumSize(400, 200)
        self.ViewList_Layout  = QtWidgets.QVBoxLayout()
        self.ViewList_Box.setLayout(self.ViewList_Layout)

        self.ViewList_Layout.setSpacing(2)
        self.ViewList_Layout.setContentsMargins(2, 2, 2, 2)

        self.generalLayout.addWidget(self.ViewList_Box)

    def chooseGrayscaleImage(self, index):
        pixmap = QPixmap.fromImage(self.rawImages[index])
        self.displayImage.setPixmap((pixmap).scaled(2000,5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseRedImage(self, index):
        self.r_image = self.rawImages[index]
        composite = create_composite_image(self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.setPixmap((pixmap).scaled(2000,5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseGreenImage(self, index):
        self.g_image = self.rawImages[index]
        composite = create_composite_image(self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.setPixmap((pixmap).scaled(2000,5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def chooseBlueImage(self, index):
        self.b_image = self.rawImages[index]
        composite = create_composite_image(self.r_image, self.g_image, self.b_image)
        pixmap = QPixmap.fromImage(composite)
        self.displayImage.setPixmap((pixmap).scaled(2000,5000, Qt.KeepAspectRatio))
        self.adjustSize()

    def _createDisplay(self):
        #Create display widget
        self.display = QScrollArea()
        self.displayImage = QLabel()

        #Set up display window properties
        self.display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.display.setWidgetResizable(True)

        self.display.setWidget(self.displayImage)
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(600, 1000)

        #add this display to the general layout
        self.generalLayout.addWidget(self.display)

    def openImages(self):
        fileNames = QFileDialog.getExistingDirectory(self, self.tr("Select image(s) to open"))
         
        (imageStack, imageNames) = self.LoadFilesOrDirectories(fileNames)
        self.setmodel(imageset.ImageSet(imageStack), imageNames) 
        #self.chooseGrayscaleImage(0)

    def LoadFilesOrDirectories(self, input):
        if isinstance(input, str):
            return self.LoadFileOrDirectory(input)
        elif isinstance(input, collections.abc.Iterable):    
            output = None
            for filename in input:
                (imageArray, fileNames) = self.LoadFileOrDirectory(filename)
                if output is None:
                    output = (imageArray, fileNames)
                else:
                    output[1].extend(fileNames)
                    output = (np.concatenate((output[0], imageArray)), output[1])
            return output
        else:
            raise ValueError("Unexpected argument type")

    def LoadFileOrDirectory(self, item):
        if os.path.isdir(item):
            output = None
            for root, dirs, files in os.walk(item):
                files = sorted(files)
                (imageArray, fileNames) = self.LoadFilesOrDirectories([os.path.join(item,f) for f in files])
                if output is None:
                    output = (imageArray, fileNames)
                else:
                    output[1].extend(fileNames)
                    output = (np.concatenate((output[0], imageArray)), output[1])
            return output
        elif os.path.isfile(item):
            return self.LoadFile(item)

    def LoadFile(self, item):
        fileName = item
        basefileName = os.path.basename(fileName)
        simpleName = os.path.splitext(basefileName)[0]
        self.fileNameList.append(simpleName)
        #self.importImageWrapper(fileName)
        #self.imageListLayout(fileName, len(self.fileNameList) -1 )

        image = nornir_imageregistration.LoadImage(item, dtype=np.float16)
        image = np.expand_dims(image,0)

        return (image,[fileName])
        
        

        #print(f'npImages Shape: {self.npImages.shape}')
            #self.colorRBs(fileName, index) 
             
        # for index in range(len(fileNames[0])):
        #     fileName = fileNames[0][index]
        #     basefileName = os.path.basename(fileName)
        #     simpleName = os.path.splitext(basefileName)[0]
        #     self.fileNameList.append(simpleName)
        #     self.importImageWrapper(fileName)
        #     self.colorRBs(fileName, index)
        

    def importImageWrapper(self, fileName):
        '''
        Imports images into UI
        :param str fileName: Filename designated by openImages
        :return: viewable image with color select radio buttons
        :rtype: QImage
        '''
        self.rawImages.append(QImage(fileName).convertToFormat(QImage.Format_RGB32)) 
 
    def createImageListEntry(self, fileName, index):
        row = QtWidgets.QGroupBox()
        rowLayout = QtWidgets.QHBoxLayout()
    
        #Add Filenames associated with RBs
        basefileName = os.path.basename(fileName)
        simpleName = os.path.splitext(basefileName)[0]
        rowLayout.addWidget(QLabel(simpleName))

         #Adding buttons for grayscale
        grayRadioButton = QRadioButton('gray')
        grayRadioButton.toggled.connect(lambda:self.chooseGrayscaleImage(index))
        rowLayout.addWidget(grayRadioButton)
        self.column1.addButton(grayRadioButton)

    #Adding buttons for red
        redRadioButton = QRadioButton("R")
        redRadioButton.toggled.connect(lambda:self.chooseRedImage(index))
        rowLayout.addWidget(redRadioButton)
        self.redRBlist.addButton(redRadioButton)

    #Adding buttons for green
        greenRadioButton = QRadioButton("G")
        greenRadioButton.toggled.connect(lambda:self.chooseGreenImage(index))
        rowLayout.addWidget(greenRadioButton)
        self.greenRBlist.addButton(greenRadioButton)

    #Adding buttons for blue
        blueRadioButton = QRadioButton("B")
        blueRadioButton.toggled.connect(lambda:self.chooseBlueImage(index))
        rowLayout.addWidget(blueRadioButton)
        self.blueRBlist.addButton(blueRadioButton)

    #add checkbox for clustering
        self.clusterCB = QCheckBox('cluster')
        self.clusterCB.stateChanged.connect(lambda:self.checkBtnState(index))
        rowLayout.addWidget(self.clusterCB)
     
        row.setLayout(rowLayout)

        return row
    
    def checkBtnState(self, index):
        self.model.selected[index] = not self.model.selected[index]

     #Select images for clustering using GUI 
#    def selectClustImages(self):
#        '''
#        Select images for clustering using GUI
#        '''
#        self.clusterview = clusterSelect(self.fileNameList)
#        self.clusterview.show()
#        self.selectedMask = self.clusterview.selectedMask
    
    def clusterKMeans(self):
        cluster = imageset.kmeansCluster(self.npImages, self.selectedMask)
        return


#Client code
def main():
    """Main function"""
    #Create QApplication instance (remember 1 per project)
    CMPViewer = QApplication(sys.argv)
    #show the UI
    view = ImageViewerUi()
    view.show()
    #Execute the main loop
    sys.exit(CMPViewer.exec_())

if __name__=='__main__':
    main()
