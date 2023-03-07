import cv2
import clusterImgSelect
from PyQt5.QtWidgets import QWidget


class Cluster(QWidget):
    def __init__(self, clusterImgName, clusterImages):
        super().__init__()
