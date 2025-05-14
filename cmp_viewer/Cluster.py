import typing
from typing import Callable, Tuple, List, Any, Dict
import collections

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtGui import QPixmap, QImage, qRgb, QColor
from sklearn.cluster import KMeans
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QInputDialog, QGraphicsPixmapItem
import cmp_viewer.models

"""
This module provides clustering functionality for multidimensional images.
It implements K-means clustering and visualization of clustered images.
"""

class KMeansSettings(typing.NamedTuple):
    """
    A named tuple for storing K-means clustering parameters.

    Attributes:
        n_clusters (int): Number of clusters to form.
        init (str): Method for initialization ('random', 'k-means++', etc.).
        n_init (int): Number of times the k-means algorithm will be run with different seeds.
        max_iter (int): Maximum number of iterations for a single run.
        tol (float): Relative tolerance for convergence.
        random_state (int): Seed for random number generation for reproducibility.
    """
    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    tol: float
    random_state: int

class Cluster(QWidget):
    """
    A widget for performing clustering on multidimensional images.

    This class provides a user interface for running K-means clustering on selected images.
    It maintains the current clustering state (labels and masks) and supports operations
    like running clustering on specific regions and undoing clustering operations.

    Attributes:
        clusterImgName: Name of the clustered image.
        _image_set (ImageSet): Set of images to cluster.
        _mask (NDArray[bool]): Boolean mask indicating which images to include in clustering.
        labels (NDArray[int]): Current cluster labels for each pixel.
        masks (Dict[int, Tuple[NDArray[bool], QColor]]): Masks and colors for each cluster.
        undo_stack (List): Stack of previous clustering states for undo operations.
        undo_stack_max_size (int): Maximum size of the undo stack.
    """

    def __init__(self, clusterImgName, clusterImages: cmp_viewer.models.ImageSet, selected_mask: NDArray[bool],
                 on_cluster_callback: Callable[[NDArray[int], Any], Tuple[NDArray[int], Any]]):
        """
        Initialize the Cluster widget.

        Args:
            clusterImgName: Name of the clustered image.
            clusterImages (ImageSet): Set of images to cluster.
            selected_mask (NDArray[bool]): Boolean mask indicating which images to include.
            on_cluster_callback (Callable): Function to call after clustering is complete.
                                           Takes labels and settings as input and returns
                                           updated labels and settings.
        """
        super().__init__()
        self.on_cluster_callback = on_cluster_callback
        self.clusterImgName = clusterImgName
        self._image_set = clusterImages
        self._mask = selected_mask
        self.labels = None  # Store current cluster labels
        self.masks = None  # Store masks and colors for each cluster: Dict[int, Tuple[NDArray[bool], QColor]]
        self.undo_stack = []  # Stack to store previous states (labels, masks)
        self.undo_stack_max_size = 10  # Limit undo stack size

        # Set up the UI
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
        """
        Run K-means clustering on the selected images.

        This method prompts the user for the number of clusters (k), performs K-means
        clustering on the selected images, and updates the labels and masks. It also
        saves the clustering state to the undo stack and calls the callback function.

        Returns:
            None
        """
        # Get user input for k
        k, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter the value of k:", 8, 1, 256)
        if not ok:
            return

        # Pixels from selected masked regions
        pixels = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)

        if pixels.size == 0:
            return

        # Create settings with default parameters and user-specified k
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
        self.labels = kmeans.labels_.reshape(self._image_set.image_shape)
        self.masks = self.generate_masks(self.labels, settings.n_clusters)

        # Save initial state to undo stack
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        # Call the callback with labels and settings
        self.on_cluster_callback(self.labels, settings)

    def generate_masks(self, labels: NDArray[int], n_clusters: int) -> Dict[int, Tuple[NDArray[bool], QColor]]:
        """
        Generate binary masks and assign a unique color for each cluster.

        This method creates a binary mask for each unique cluster label and assigns
        a visually distinct color to each cluster using a hue-based approach.

        Args:
            labels (NDArray[int]): Array of cluster labels for each pixel.
            n_clusters (int): Number of clusters (may be different from actual unique labels).

        Returns:
            Dict[int, Tuple[NDArray[bool], QColor]]: Dictionary mapping cluster IDs to
                                                    tuples of (binary mask, color).
        """
        masks = {}
        unique_labels = np.unique(labels)
        for idx, cluster_id in enumerate(unique_labels):
            # Create binary mask for this cluster (True where label matches cluster_id)
            mask = (labels == cluster_id)

            # Generate a unique color for each cluster using a hue-based approach
            hue = (idx * 137.5) % 360  # Golden angle approximation for distinct hues
            color = QColor.fromHsv(int(hue), 200, 200)  # High saturation and value for vibrant colors

            masks[cluster_id] = (mask, color)
        return masks

    def cluster_on_mask(self, mask: NDArray[bool], n_clusters: int) -> Tuple[NDArray[int], KMeansSettings]:
        """
        Run k-means clustering on pixels within the given mask using averaged image data.

        This method performs K-means clustering on a specific region of the image defined
        by the mask. It averages pixel values across selected images within the masked region,
        runs K-means on these averaged values, and updates the labels and masks accordingly.

        Args:
            mask (NDArray[bool]): Boolean mask indicating which pixels to include in clustering.
            n_clusters (int): Number of clusters to form within the masked region.

        Returns:
            Tuple[NDArray[int], KMeansSettings]: Tuple containing the new labels array and
                                               the settings used for clustering. Returns
                                               (None, None) if clustering cannot be performed.

        Raises:
            ValueError: If the mask shape does not match the image dimensions.
        """
        # Check if there are images to cluster or if the mask is empty
        if self._image_set.images.size == 0 or not np.any(mask):
            return None, None

        # Get the selected images
        selected_images = self._image_set.images[self._mask, :, :]
        if selected_images.size == 0:
            return None, None

        # Ensure mask matches the image dimensions (height, width)
        if mask.shape != selected_images.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} does not match image dimensions {selected_images.shape[1:]}")

        # Average pixel values across selected images within the mask
        masked_images = [img[mask] for img in selected_images]
        if not masked_images or all(len(m) == 0 for m in masked_images):
            return None, None
        avg_masked_pixels = np.mean(np.vstack(masked_images), axis=0)
        avg_masked_pixels = avg_masked_pixels.reshape(-1, 1)

        # Debug: Verify sizes
        print(f"Number of masked pixels: {avg_masked_pixels.shape[0]}")
        print(f"Number of True values in mask: {np.sum(mask)}")

        # Run k-means on averaged masked pixels
        settings = KMeansSettings(n_clusters=n_clusters, init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42)
        kmeans = KMeans(n_clusters=settings.n_clusters,
                        init=settings.init,
                        n_init=settings.n_init,
                        max_iter=settings.max_iter,
                        tol=settings.tol,
                        random_state=settings.random_state)
        sub_labels = kmeans.fit_predict(avg_masked_pixels)

        # Debug: Verify sub_labels size
        print(f"Sub_labels size: {sub_labels.shape[0]}")

        # Create new labels array, preserving original labels outside the mask
        new_labels = np.copy(self.labels)
        max_label = np.max(self.labels) if self.labels is not None else -1
        new_labels[mask] = sub_labels + max_label + 1  # Offset new labels to avoid overlap

        # Update masks with the new labels
        self.labels = new_labels
        self.masks = self.generate_masks(self.labels, len(np.unique(new_labels)))

        # Save state to undo stack
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        return new_labels, settings

    def undo_clustering(self):
        """
        Revert to the previous clustering state.

        This method restores the previous clustering state from the undo stack,
        updating the labels and masks accordingly. It then calls the callback
        function with the restored labels and default settings.

        Returns:
            bool: True if the undo operation was successful, False if there are
                 no previous states to revert to.
        """
        # Ensure there's at least one state remaining after the undo
        if len(self.undo_stack) <= 1:  # At least one state must remain
            return False

        self.undo_stack.pop()  # Remove current state
        prev_labels, prev_masks = self.undo_stack[-1]  # Get previous state

        # Restore previous state
        self.labels = np.copy(prev_labels)
        self.masks = prev_masks

        # Call callback with restored labels and default settings based on number of unique labels
        self.on_cluster_callback(self.labels, KMeansSettings(
            n_clusters=len(np.unique(self.labels)), 
            init="random", 
            n_init=5, 
            max_iter=100, 
            tol=1e-3, 
            random_state=42
        ))

        return True
