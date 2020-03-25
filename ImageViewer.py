# Filename: ImageViewer.py
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
import sys
import os

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
__author__ = "RL Pfeiffer"

#QMainwindow subclass for calc GUI

class ImageViewerUi(QMainWindow):
    rawImages = []
    fileNameList = []
    grayRBlist = []
    redRBlist = []
    greenRBlist = []
    blueRBlist = []
    #redImageIndex = -1
    # greenImageIndex = -1
    # blueImageIndex = -1

    """View Gui"""
    def __init__(self):
        """View Initializer"""
        super().__init__()
        #Some main properties of the WindowsError
        self.setWindowTitle('ImageViewer')

        #Set central widget and the general layout
        self.generalLayout = QHBoxLayout()
        self._centralWidget = QScrollArea()
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        self._centralWidget.setMinimumSize(2000, 800)


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

    def chooseDisplayImage(self, index):
        pixmap = QPixmap.fromImage(self.rawImages[index])
        self.display.setPixmap(pixmap)
        self.resize(pixmap.size())
        self.adjustSize()

    def chooseGrayscaleImage(self, index):
        self.chooseDisplayImage(self.fileNameList[index])

    def _createDisplay(self):
        #Create display widget
        self.display = QLabel()
        #set display properties
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(600, 1000)

        #add this display to the general layout
        self.generalLayout.addWidget(self.display)

    def openImages(self):
        fileNames = QFileDialog.getOpenFileNames(self, self.tr("Select image(s) to open"))
        for fileName in fileNames[0]:
            self.importImageWrapper(fileName)
            self.colorRBs(fileName)
            self.fileNameList.append(fileName)
        self.chooseDisplayImage(0)

    def importImageWrapper(self, fileName):
        self.rawImages.append(QImage(fileName))

    def colorRBs(self, fileName):
        row = QtWidgets.QGroupBox()
        rowLayout = QtWidgets.QHBoxLayout()

    #Adding buttons for grayscale
        grayRadioButton = QRadioButton("Gray")
        grayRadioButton.toggled.connect(self.chooseGrayscaleImage)
        rowLayout.addWidget(grayRadioButton)
        self.grayRBlist.append(grayRadioButton)

    #Adding buttons for red
        redRadioButton = QRadioButton("R")
        #redRadioButton.toggled.connect(self.redImageSelector)
        rowLayout.addWidget(redRadioButton)
        self.redRBlist.append(redRadioButton)

    #Adding buttons for green
        greenRadioButton = QRadioButton("G")
        #greenRadioButton.toggled.connect(self.greenImageSelector)
        rowLayout.addWidget(greenRadioButton)
        self.greenRBlist.append(greenRadioButton)

    #Adding buttons for blue
        blueRadioButton = QRadioButton("B")
        #blueRadioButton.toggled.connect(self.greenImageSelector)
        rowLayout.addWidget(blueRadioButton)
        self.blueRBlist.append(blueRadioButton)

    #Add Filenames associated with RBs
        basefileName = os.path.basename(fileName)
        rowLayout.addWidget(QLabel(basefileName))

        row.setLayout(rowLayout)
        self.ViewList_Layout.addWidget(row)


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
