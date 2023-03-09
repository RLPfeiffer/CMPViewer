import typing
from typing import Callable, Tuple, List, Any
import collections

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtGui import QPixmap, QImage, qRgb
from sklearn.cluster import KMeans
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QInputDialog, QGraphicsPixmapItem
import cmp_viewer.models

class KMeansSettings(typing.NamedTuple):
    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    tol: float
    random_state: int

class Cluster(QWidget):

    def __init__(self, clusterImgName, clusterImages: cmp_viewer.models.ImageSet, selected_mask: NDArray[bool],
                 on_cluster_callback: Callable[[...], Tuple[NDArray[int], Any]]):
        super().__init__()
        self.on_cluster_callback = on_cluster_callback
        self.clusterImgName = clusterImgName
        self._image_set = clusterImages
        self._mask = selected_mask

        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        self.clusterList.addItems(["k-means"])
        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Run Clustering")
        self.button1.clicked.connect(self.runKMeansClustering)

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def runKMeansClustering(self):
        # Get user input for k
        k, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter the value of k:", 8, 1, 256)
        if not ok:
            return

        #pixels = self.clusterImages.reshape((-1, 1))
        pixels = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)

        # Perform k-means clustering on each image in the list
        #for i, img in enumerate(self.clusterImages):
            # Reshape image into 2D array
        #    pixels = img.reshape((-1, 1))

        # Convert to float32 for k-means function
        # pixels = np.float32(pixels)
        settings = KMeansSettings(n_clusters=k, init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=settings.n_clusters,
                        init=settings.init,
                        n_init=settings.n_init,
                        max_iter=settings.max_iter,
                        tol=settings.tol,
                        random_state=settings.random_state)
        kmeans.fit(pixels.T)

        # Get the label array and reshape it to match the original image shape
        labels = kmeans.labels_.reshape(self._image_set.image_shape)

        self.on_cluster_callback(labels, settings)




