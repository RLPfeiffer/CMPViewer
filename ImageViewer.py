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
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from functools import partial

__version__ = '1.0'
__author__ = "RL Pfeiffer"

#QMainwindow subclass for calc GUI

class ImageViewerUi(QMainWindow):
    rawImages = []
    grayRB = None
    #redRB = QRadioButton(None)
    #greenRB = QRadioButton(None)
    #blueRB = QRadioButton(None)
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

        if self.grayRB is None:
            pass
        else:
            self.grayRB.toggled.connect(self.changeGrayscaleImage)


    def chooseDisplayedImage(self, index):
        pixmap = QPixmap.fromImage(self.rawImages[index])
        self.display.setPixmap(pixmap)
        self.resize(pixmap.size())
        self.adjustSize()

    def changeGrayscaleImage(self, imageItem):
        self.chooseDisplayedImage(imageItem.index().row())

        # print(type(image))
        # print(image)
        # if type(image) is QPixmap:
        #      pixmap = image
        # elif type(image) is QImage:
        #      pixmap = QPixmap.fromImage(image)
        # else:
        #     raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        # if self.hasImage():
        #     self.display.setPixmap(pixmap)
        #
        # self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        # self._createDisplay()

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
        self.chooseDisplayedImage(0)

    def importImageWrapper(self, fileName):
        self.rawImages.append(QImage(fileName))

        row = QtWidgets.QGroupBox()
        rowLayout = QtWidgets.QHBoxLayout()

        self.grayRB = QRadioButton("Gray")
        rowLayout.addWidget(self.grayRB)
        self.redRB = QRadioButton("R")
        rowLayout.addWidget(self.redRB)
        self.greenRB = QRadioButton("G")
        rowLayout.addWidget(self.greenRB)
        self.blueRB = QRadioButton("B")
        rowLayout.addWidget(self.blueRB)
        rowLayout.addWidget(QLabel(fileName))

        row.setLayout(rowLayout)
        self.ViewList_Layout.addWidget(row)

#Client code
def main():
    """Main function"""
    #Create QApplication instance (remember 1 per project)
    pycalc = QApplication(sys.argv)
    #show the UI
    view = ImageViewerUi()
    view.show()
    #Execute the main loop
    sys.exit(pycalc.exec_())

if __name__=='__main__':
    main()
