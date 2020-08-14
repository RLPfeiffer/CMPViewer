# Filename: ImageViewer.py
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
import sys
import os
from rgb import *

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
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from functools import partial

__version__ = '1.0'
__author__ = "RL Pfeiffer & NQN Studios"

#QMainwindow subclass for calc GUI

class ImageViewerUi(QMainWindow):
    rawImages = []
    fileNameList = []
    r_image = None
    g_image = None
    b_image = None
    #redImageIndex = -1
    # greenImageIndex = -1
    # blueImageIndex = -1

    """View Gui"""
    def __init__(self):
        """View Initializer"""
        super().__init__()
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

    def _createMenuBar(self):
        """Create a menubar"""
        menuBar = self.menuBar()
        self.menuBar = QMenuBar()
        menuBar.setNativeMenuBar(False)
        fileMenu = menuBar.addMenu('File')
        self.generalLayout.addWidget(self.menuBar)

        """create menu items"""
        openAct = QAction('Open Images', self)
        openAct.triggered.connect(self.openImages)
        fileMenu.addAction(openAct)

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
        fileNames = QFileDialog.getOpenFileNames(self, self.tr("Select image(s) to open"))
        self.ImportLayout = QVBoxLayout()
        self.column1 = QtWidgets.QButtonGroup()
        self.redRBlist = QtWidgets.QButtonGroup()
        self.greenRBlist = QtWidgets.QButtonGroup()
        self.blueRBlist = QtWidgets.QButtonGroup()
        self.ViewList_Layout.addLayout(self.ImportLayout)
        for index in range(len(fileNames[0])):
            fileName = fileNames[0][index]
            self.importImageWrapper(fileName)
            self.colorRBs(fileName, index)
            self.fileNameList.append(fileName)
        self.chooseGrayscaleImage(0)

    def importImageWrapper(self, fileName):
        self.rawImages.append(QImage(fileName).convertToFormat(QImage.Format_RGB32))

    def colorRBs(self, fileName, index):
        row = QtWidgets.QGroupBox()
        rowLayout = QtWidgets.QHBoxLayout()
        #column1 = QtWidgets.QButtonGroup()

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

        row.setLayout(rowLayout)
        self.ImportLayout.addWidget(row)




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
