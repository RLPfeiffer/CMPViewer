import sys

from PyQt5.QtWidgets import QApplication

from cmp_viewer.ImageViewer import ImageViewerUi
from cmp_viewer import ImageViewer

def main():
    """Main function"""
    # Create QApplication instance (remember 1 per project)
    CMPViewer = QApplication(sys.argv)
    # show the UI
    view = ImageViewerUi()
    view.show()
    # Execute the main loop
    sys.exit(CMPViewer.exec_())

if __name__ == '__main__':
    main()