import sys
from PyQt5.QtWidgets import QApplication
from cmp_viewer.ImageViewer import ImageViewerUi

def main():
    """Main function"""
    CMPViewer = QApplication(sys.argv)
    view = ImageViewerUi()
    view.show()
    sys.exit(CMPViewer.exec_())

if __name__ == '__main__':
    main()
