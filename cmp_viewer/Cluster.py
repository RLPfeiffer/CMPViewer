import typing
from typing import Callable, Tuple, List, Any, Dict
import collections
import os
import cv2

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtGui import QPixmap, QImage, qRgb, QColor
from sklearn.cluster import KMeans
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QInputDialog, QGraphicsPixmapItem
import cmp_viewer.models
import cmp_viewer.utils

"""
This module provides clustering functionality for multidimensional images.
It implements K-means clustering, ISODATA clustering, and visualization of clustered images.
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

class ISODATASettings(typing.NamedTuple):
    """
    A named tuple for storing ISODATA clustering parameters.

    ISODATA (Iterative Self-Organizing Data Analysis Technique) is an extension
    of k-means that allows for merging and splitting of clusters based on various criteria.

    Attributes:
        n_clusters (int): Initial number of clusters to form.
        max_iter (int): Maximum number of iterations.
        min_samples (int): Minimum number of samples in a cluster.
        max_std_dev (float): Maximum standard deviation within a cluster.
        min_cluster_distance (float): Minimum distance between clusters for merging.
        max_merge_pairs (int): Maximum number of cluster pairs to merge per iteration.
        random_state (int): Seed for random number generation for reproducibility.
    """
    n_clusters: int
    max_iter: int
    min_samples: int
    max_std_dev: float
    min_cluster_distance: float
    max_merge_pairs: int
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
        self.clusterList.addItems(["k-means", "ISODATA"])
        self.clusterList.setCurrentRow(0)  # Default to k-means
        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Run Clustering")
        self.button1.clicked.connect(self.runClustering)

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def runClustering(self):
        """
        Run the selected clustering algorithm on the selected images.

        This method determines which clustering algorithm to run based on the
        selected item in the clusterList, and then calls the appropriate method.

        Returns:
            None
        """
        selected_algorithm = self.clusterList.currentItem().text()
        if selected_algorithm == "k-means":
            self.runKMeansClustering()
        elif selected_algorithm == "ISODATA":
            self.runISODATAClustering()
        else:
            print(f"Unknown clustering algorithm: {selected_algorithm}")

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

    def runISODATAClustering(self):
        """
        Run ISODATA clustering on the selected images.

        ISODATA (Iterative Self-Organizing Data Analysis Technique) is an extension
        of k-means that allows for merging and splitting of clusters based on various criteria.
        This method prompts the user for the necessary parameters, performs ISODATA
        clustering on the selected images, and updates the labels and masks. It also
        saves the clustering state to the undo stack and calls the callback function.

        Returns:
            None
        """
        # Get user input for parameters
        k, ok = QInputDialog.getInt(self, "ISODATA Clustering", "Initial number of clusters:", 8, 1, 256)
        if not ok:
            return

        max_iter, ok = QInputDialog.getInt(self, "ISODATA Clustering", "Maximum iterations:", 20, 1, 1000)
        if not ok:
            return

        min_samples, ok = QInputDialog.getInt(self, "ISODATA Clustering", "Minimum samples per cluster:", 5, 1, 1000)
        if not ok:
            return

        max_std_dev, ok = QInputDialog.getDouble(self, "ISODATA Clustering", "Maximum standard deviation:", 1.0, 0.1, 10.0, 1)
        if not ok:
            return

        min_cluster_distance, ok = QInputDialog.getDouble(self, "ISODATA Clustering", "Minimum cluster distance:", 2.0, 0.1, 10.0, 1)
        if not ok:
            return

        max_merge_pairs, ok = QInputDialog.getInt(self, "ISODATA Clustering", "Maximum merge pairs:", 2, 1, 10)
        if not ok:
            return

        # Get the image shape for later reshaping
        height, width = self._image_set.image_shape

        # Get the pixel values for all selected images
        # This selects the images indicated by self._mask (which is a 1D array)
        # and reshapes them to a 2D array where each row is a flattened image
        pixel_values = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)

        if pixel_values.size == 0:
            return

        # Create settings with user-specified parameters
        settings = ISODATASettings(
            n_clusters=k,
            max_iter=max_iter,
            min_samples=min_samples,
            max_std_dev=max_std_dev,
            min_cluster_distance=min_cluster_distance,
            max_merge_pairs=max_merge_pairs,
            random_state=42
        )

        # Use the ISODATA algorithm
        # _isodata_algorithm expects data with shape (n_features, n_samples)
        # In our case, n_features is n_selected_images and n_samples is height*width
        cluster_labels = self._isodata_algorithm(pixel_values, settings)

        # Reshape the cluster labels to match the image shape
        self.labels = cluster_labels.reshape(height, width)
        self.masks = self.generate_masks(self.labels, len(np.unique(self.labels)))

        # Save initial state to undo stack
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        # Call the callback with labels and settings
        self.on_cluster_callback(self.labels, settings)

    def _isodata_algorithm(self, data: NDArray, settings: ISODATASettings) -> NDArray[int]:
        """
        Implement the ISODATA clustering algorithm.

        ISODATA (Iterative Self-Organizing Data Analysis Technique) is an extension
        of k-means that allows for merging and splitting of clusters based on various criteria.

        Args:
            data (NDArray): Data to cluster, shape (n_features, n_samples)
            settings (ISODATASettings): Settings for the ISODATA algorithm

        Returns:
            NDArray[int]: Cluster labels for each sample
        """
        np.random.seed(settings.random_state)
        n_samples = data.shape[1]
        n_features = data.shape[0]

        # Adjust number of clusters if it exceeds number of samples
        settings = settings._replace(n_clusters=min(settings.n_clusters, n_samples))

        # Initialize centroids randomly
        # Select k random samples as initial centroids
        indices = np.random.choice(n_samples, settings.n_clusters, replace=False)
        centroids = data[:, indices]

        # Initialize labels
        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(settings.max_iter):
            # Store current number of clusters before any modifications
            old_n_clusters = settings.n_clusters

            # Assign samples to closest centroids (like k-means)
            distances = np.zeros((settings.n_clusters, n_samples))
            for i in range(settings.n_clusters):
                diff = data - centroids[:, i].reshape(-1, 1)
                distances[i] = np.sum(diff**2, axis=0)

            # Assign each sample to the closest centroid
            labels = np.argmin(distances, axis=0)

            # Make a copy of the current centroids for convergence check
            old_centroids = centroids.copy()

            # Update centroids based on new assignments
            for i in range(settings.n_clusters):
                cluster_samples = data[:, labels == i]
                if cluster_samples.shape[1] > 0:
                    centroids[:, i] = np.mean(cluster_samples, axis=1)

            # Check for empty clusters and handle them
            for i in range(settings.n_clusters):
                if np.sum(labels == i) == 0:
                    # Find the cluster with the most samples
                    largest_cluster = np.argmax([np.sum(labels == j) for j in range(settings.n_clusters)])
                    # Find the samples furthest from the centroid in the largest cluster
                    cluster_samples = data[:, labels == largest_cluster]
                    if cluster_samples.shape[1] > 0:
                        diff = cluster_samples - centroids[:, largest_cluster].reshape(-1, 1)
                        distances = np.sum(diff**2, axis=0)
                        furthest_sample_idx = np.argmax(distances)
                        # Set the empty cluster's centroid to this sample
                        centroids[:, i] = cluster_samples[:, furthest_sample_idx]
                        # Reassign some samples to this new centroid
                        diff = data - centroids[:, i].reshape(-1, 1)
                        new_distances = np.sum(diff**2, axis=0)
                        closest_to_new = np.argsort(new_distances)[:settings.min_samples]
                        labels[closest_to_new] = i

            # ISODATA specific steps:

            # 1. Discard clusters with too few samples
            for i in range(settings.n_clusters):
                if np.sum(labels == i) < settings.min_samples:
                    # Reassign samples from small clusters to the closest remaining cluster
                    small_cluster_samples = np.where(labels == i)[0]
                    for sample_idx in small_cluster_samples:
                        # Find the next closest centroid
                        sample = data[:, sample_idx]
                        distances = np.array([np.sum((sample - centroids[:, j])**2) for j in range(settings.n_clusters) if j != i])
                        closest_centroid = np.argmin(distances)
                        # Adjust for the removed index
                        if closest_centroid >= i:
                            closest_centroid += 1
                        labels[sample_idx] = closest_centroid

                    # Remove the centroid
                    centroids = np.delete(centroids, i, axis=1)

                    # Update labels to reflect the removed centroid
                    labels[labels > i] -= 1

                    # Adjust the number of clusters
                    settings = settings._replace(n_clusters=settings.n_clusters - 1)

                    # Break to recalculate everything with the new number of clusters
                    break

            # 2. Split clusters with large standard deviation
            for i in range(settings.n_clusters):
                cluster_samples = data[:, labels == i]
                if cluster_samples.shape[1] > 2 * settings.min_samples:
                    # Calculate standard deviation of the cluster
                    std_dev = np.std(cluster_samples, axis=1)

                    # If any dimension has std dev greater than the threshold, split the cluster
                    if np.any(std_dev > settings.max_std_dev):
                        # Add a new centroid
                        new_centroid_idx = settings.n_clusters
                        settings = settings._replace(n_clusters=settings.n_clusters + 1)

                        # Find the dimension with the largest std dev
                        max_std_dim = np.argmax(std_dev)

                        # Create two new centroids by moving along this dimension
                        new_centroids = np.column_stack((
                            centroids,
                            centroids[:, i].copy()
                        ))

                        # Adjust the centroids along the dimension with largest variance
                        new_centroids[max_std_dim, i] -= std_dev[max_std_dim]
                        new_centroids[max_std_dim, new_centroid_idx] += std_dev[max_std_dim]

                        centroids = new_centroids

                        # Reassign samples to the new centroids
                        diff1 = data - centroids[:, i].reshape(-1, 1)
                        diff2 = data - centroids[:, new_centroid_idx].reshape(-1, 1)
                        dist1 = np.sum(diff1**2, axis=0)
                        dist2 = np.sum(diff2**2, axis=0)

                        # Assign to the closer of the two centroids
                        labels[np.logical_and(labels == i, dist2 < dist1)] = new_centroid_idx

                        # Break to recalculate everything with the new number of clusters
                        break

            # 3. Merge clusters that are close to each other
            if settings.n_clusters >= 2:
                # Calculate distances between all pairs of centroids
                centroid_distances = np.zeros((settings.n_clusters, settings.n_clusters))
                for i in range(settings.n_clusters):
                    for j in range(i+1, settings.n_clusters):
                        centroid_distances[i, j] = np.sqrt(np.sum((centroids[:, i] - centroids[:, j])**2))
                        centroid_distances[j, i] = centroid_distances[i, j]

                # Find pairs of clusters to merge (closest pairs first)
                merge_candidates = []
                for i in range(settings.n_clusters):
                    for j in range(i+1, settings.n_clusters):
                        if centroid_distances[i, j] < settings.min_cluster_distance:
                            merge_candidates.append((i, j, centroid_distances[i, j]))

                # Sort by distance (closest first)
                merge_candidates.sort(key=lambda x: x[2])

                # Merge up to max_merge_pairs pairs
                merged_clusters = set()
                for i, j, _ in merge_candidates[:settings.max_merge_pairs]:
                    if i in merged_clusters or j in merged_clusters:
                        continue

                    # Merge clusters i and j
                    # Calculate the weighted average of the centroids
                    ni = np.sum(labels == i)
                    nj = np.sum(labels == j)

                    if ni == 0 or nj == 0:
                        continue

                    new_centroid = (ni * centroids[:, i] + nj * centroids[:, j]) / (ni + nj)

                    # Update centroid i with the merged centroid
                    centroids[:, i] = new_centroid

                    # Reassign samples from cluster j to cluster i
                    labels[labels == j] = i

                    # Mark cluster j as merged
                    merged_clusters.add(j)

                # Remove merged centroids
                if merged_clusters:
                    # Convert to list and sort in descending order to avoid index issues
                    merged_list = sorted(list(merged_clusters), reverse=True)
                    for idx in merged_list:
                        centroids = np.delete(centroids, idx, axis=1)
                        # Update labels to reflect the removed centroid
                        for old_idx in range(idx, settings.n_clusters):
                            labels[labels == old_idx] = old_idx - 1

                    # Update the number of clusters
                    settings = settings._replace(n_clusters=settings.n_clusters - len(merged_clusters))

            # Check for convergence only if the number of clusters hasn't changed
            if old_n_clusters == settings.n_clusters:
                if np.allclose(old_centroids[:, :settings.n_clusters], centroids):
                    break
            # If number of clusters changed, continue to next iteration
            else:
                continue

        # Ensure labels are consecutive integers starting from 0
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        new_labels = np.array([label_map[l] for l in labels])

        return new_labels

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

    def create_label_image(self, labels: NDArray[int], num_labels: int) -> Image.Image:
        """
        Create a PIL Image from cluster labels.

        Args:
            labels (NDArray[int]): Array of cluster labels
            num_labels (int): Number of unique labels

        Returns:
            Image.Image: PIL Image representing the clustered data
        """
        return cmp_viewer.utils.numpy_labels_to_pillow_image(labels)

    def create_mask_overlay(self, mask: NDArray[bool], color: QColor, opacity: int, 
                           target_width: int = None, target_height: int = None) -> QImage:
        """
        Create a QImage overlay for a cluster mask with specified color and opacity.

        Args:
            mask (NDArray[bool]): Boolean mask for the cluster
            color (QColor): Color to use for the mask
            opacity (int): Opacity value (0-255)
            target_width (int, optional): Target width for resizing
            target_height (int, optional): Target height for resizing

        Returns:
            QImage: Transparent overlay with the mask colored
        """
        height, width = mask.shape

        # Resize mask if target dimensions are provided
        if target_width is not None and target_height is not None:
            mask_small = cv2.resize(mask.astype(np.uint8), (target_width, target_height), 
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            width, height = target_width, target_height
        else:
            mask_small = mask

        # Create transparent overlay
        overlay = QImage(width, height, QImage.Format_ARGB32)
        overlay.fill(Qt.transparent)

        # Apply color to mask
        mask_data = np.zeros((height, width, 4), dtype=np.uint8)
        mask_data[mask_small, 0] = color.red()
        mask_data[mask_small, 1] = color.green()
        mask_data[mask_small, 2] = color.blue()
        mask_data[mask_small, 3] = opacity

        # Convert to QImage
        overlay_data = mask_data.tobytes()
        overlay = QImage(overlay_data, width, height, QImage.Format_ARGB32)

        return overlay

    def export_cluster_mask(self, cluster_id: int, output_path: str, file_format: str = "tiff"):
        """
        Export a single cluster mask to a file.

        Args:
            cluster_id (int): ID of the cluster to export
            output_path (str): Path to save the mask
            file_format (str, optional): File format (tiff, png, etc.)

        Returns:
            bool: True if export was successful, False otherwise
        """
        if self.masks is None or cluster_id not in self.masks:
            return False

        mask, _ = self.masks[cluster_id]
        if mask is None:
            return False

        # Convert boolean mask to uint8 (0 or 255)
        mask_array = mask.astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode='L')

        # Ensure output path has correct extension
        if not output_path.lower().endswith(f".{file_format.lower()}"):
            output_path = f"{output_path}.{file_format.lower()}"

        mask_image.save(output_path)
        return True

    def calculate_optimal_scale_factor(self, height: int, width: int, max_pixels: int = 500000) -> float:
        """
        Calculate optimal scale factor to resize an image to a maximum number of pixels.

        Args:
            height (int): Original height
            width (int): Original width
            max_pixels (int, optional): Maximum number of pixels in the resized image

        Returns:
            float: Scale factor to apply
        """
        if height * width <= max_pixels:
            return 1.0

        return np.sqrt(max_pixels / (height * width))

    def create_color_table(self, num_labels: int) -> List[int]:
        """
        Create a color table for visualizing cluster labels.

        Args:
            num_labels (int): Number of unique labels

        Returns:
            List[int]: List of RGB values as integers
        """
        return [qRgb(int((i/num_labels) * 255), int((i/num_labels) * 255), int((i/num_labels-1) * 255)) 
                for i in range(num_labels)]

    def create_palette_from_color_table(self, color_table: List[int]) -> List[int]:
        """
        Create a palette from a color table for use with PIL images.

        Args:
            color_table (List[int]): List of RGB values as integers

        Returns:
            List[int]: Flattened list of RGB values for PIL palette
        """
        palette = []
        for rgb in color_table:
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            palette.extend([r, g, b])
        return palette

    def prepare_label_image_for_display(self, img: Image.Image, num_labels: int) -> Tuple[Image.Image, List[int]]:
        """
        Prepare a label image for display by setting its palette.

        Args:
            img (Image.Image): PIL Image with label data
            num_labels (int): Number of unique labels

        Returns:
            Tuple[Image.Image, List[int]]: Tuple containing the prepared image and color table
        """
        # Create color table and palette
        color_table = self.create_color_table(num_labels)
        palette = self.create_palette_from_color_table(color_table)

        # Convert image to palette mode if needed
        if img.mode != 'P':
            img = img.convert('P')

        # Set the palette
        img.putpalette(palette)

        return img, color_table
