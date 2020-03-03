# Filename: ImageViewer.py
"""ImageViewer is an initial core for opening and viewing CMP image stacks"""
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMenuBar
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
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
        self.ViewList_Layout  = QtWidgets.QFormLayout()
        self.ViewList_RawList = QtWidgets.QListView()
        self.ViewList_Items = QtGui.QStandardItemModel(self.ViewList_RawList)

        self.ViewList_Box.setLayout(self.ViewList_Layout)

        self.ViewList_Layout.addWidget(self.ViewList_RawList)
        self.ViewList_Layout.setSpacing(2)
        self.ViewList_Layout.setContentsMargins(2, 2, 2, 2)
        self.ViewList_RawList.setModel(self.ViewList_Items)

        self.generalLayout.addWidget(self.ViewList_Box)

        self.ViewList_Items.itemChanged.connect(self.updateDisplay)

    def updateDisplay(self, image):
        if type(image) is QPixmap:
             pixmap = image
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self._createDisplay()

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

    def importImageWrapper(self, fileName):
        image= QPixmap(fileName)
        file = QtGui.QStandardItem(fileName)
        file.setCheckable(True)
        self.ViewList_Items.appendRow(file)
        self.display.setPixmap(image)
        self.resize(image.size())
        self.adjustSize()

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
