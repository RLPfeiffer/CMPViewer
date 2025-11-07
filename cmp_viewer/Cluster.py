import typing
from typing import Callable, Tuple, List, Any, Dict
import collections
import os
import threading
import cv2

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtGui import QPixmap, QImage, qRgb, QColor
from sklearn.cluster import KMeans
from PIL import Image
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QInputDialog, QGraphicsPixmapItem, QProgressDialog, QMessageBox
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

class ClusteringWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)  # labels (ndarray), settings
    error = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, *, algorithm: str, data: np.ndarray, settings: typing.Any, image_shape: typing.Tuple[int, int], isodata_fn: typing.Callable = None):
        super().__init__()
        self.algorithm = algorithm
        self.data = data
        self.settings = settings
        self.image_shape = image_shape
        self.isodata_fn = isodata_fn
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            if self.algorithm == 'kmeans':
                self.progress.emit(0, 'Initializing k-means...')
                # sklearn expects samples as rows; upstream provides pixels as (n_selected_images, n_pixels)
                pixels = self.data
                km = KMeans(n_clusters=self.settings.n_clusters,
                            init=self.settings.init,
                            n_init=self.settings.n_init,
                            max_iter=self.settings.max_iter,
                            tol=self.settings.tol,
                            random_state=self.settings.random_state)
                # Indeterminate: cannot report inner progress
                labels = None
                km.fit(pixels.T)
                if self._cancel_event.is_set():
                    self.canceled.emit()
                    return
                labels = km.labels_.reshape(self.image_shape)
                self.progress.emit(100, 'k-means complete')
                self.finished.emit(labels, self.settings)
            elif self.algorithm == 'isodata':
                if self.isodata_fn is None:
                    raise RuntimeError('ISODATA function not provided')
                def cb(pct: int, msg: str):
                    self.progress.emit(pct, msg)
                labels_flat = self.isodata_fn(self.data, self.settings, progress_cb=cb, cancel_event=self._cancel_event)
                if labels_flat is None:
                    # Treat None as canceled
                    self.canceled.emit()
                    return
                labels = labels_flat.reshape(self.image_shape)
                self.progress.emit(100, 'ISODATA complete')
                self.finished.emit(labels, self.settings)
            else:
                raise RuntimeError(f'Unknown algorithm: {self.algorithm}')
        except Exception as e:
            # If cancellation expressed via exception, map to canceled
            msg = str(e)
            if 'CANCELED' in msg.upper() or isinstance(e, KeyboardInterrupt):
                self.canceled.emit()
            else:
                self.error.emit(msg)

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
                 on_cluster_callback: Callable[[NDArray[int], Any], Tuple[NDArray[int], Any]],
                 *, base_labels: NDArray[int] | None = None, base_masks: dict | None = None, spatial_roi: NDArray[bool] | None = None):
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
        # Prior state and ROI
        self.labels = base_labels if base_labels is not None else None  # Current cluster labels
        self.masks = base_masks if base_masks is not None else None  # Dict[int, Tuple[NDArray[bool], QColor]]
        self._spatial_roi = spatial_roi if spatial_roi is not None else None  # Optional ROI mask
        self.undo_stack = []  # Stack to store previous states (labels, masks)
        self.undo_stack_max_size = 10  # Limit undo stack size
        # Track whether the last run used an ROI (for viewer naming/clearing logic)
        self._last_run_was_roi = False
        # If prior state exists, seed undo stack
        if self.labels is not None and self.masks is not None:
            try:
                self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
            except Exception:
                # Fallback: do not seed if shapes/types unexpected
                pass

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
        The window is hidden before running the algorithm and closed after it completes.

        Returns:
            None
        """
        selected_algorithm = self.clusterList.currentItem().text()
        # Optionally hide the launcher window while clustering runs
        # (we keep it alive to own the progress dialog)
        try:
            self.hide()
        except Exception:
            pass

        if selected_algorithm == "k-means":
            self.runKMeansClustering()
        elif selected_algorithm == "ISODATA":
            self.runISODATAClustering()
        else:
            print(f"Unknown clustering algorithm: {selected_algorithm}")

    def runKMeansClustering(self):
        """
        Run K-means clustering on the selected images.

        If a spatial ROI and prior labels are available, only re-cluster pixels within
        the ROI and preserve labels outside the ROI. Otherwise, run full-image k-means.

        Returns:
            None
        """
        # Get user input for k
        k, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter the value of k:", 8, 1, 256)
        if not ok:
            return

        # Path 1: ROI-based re-clustering if prior labels and ROI are available
        if self._spatial_roi is not None and self.labels is not None:
            try:
                new_labels, settings = self.cluster_on_mask(self._spatial_roi, n_clusters=k)
                if new_labels is None:
                    QMessageBox.warning(self, "ROI Empty", "No pixels in the selected ROI. Please choose a different mask or run full-image clustering.")
                    return
                # Mark this run as ROI-based for downstream naming logic
                self._last_run_was_roi = True
                # Callback will update masks and main view
                self.on_cluster_callback(new_labels, settings)
                return
            except Exception as e:
                QMessageBox.warning(self, "ROI Clustering Error", f"Falling back to full-image k-means due to error: {e}")
                # fall-through to full-image path

        # Path 2: Full-image k-means
        # Mark as non-ROI run
        self._last_run_was_roi = False
        pixels = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)
        if pixels.size == 0:
            return

        # Create settings with default parameters and user-specified k
        settings = KMeansSettings(n_clusters=k, init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42)

        # Start background worker with progress dialog (indeterminate for k-means)
        self._start_clustering_worker(
            algorithm='kmeans',
            data=pixels,
            settings=settings,
            image_shape=self._image_set.image_shape,
            dialog_title="Running k-means...",
            determinate=False,
            maximum=settings.max_iter
        )

    def runISODATAClustering(self):
        """
        Run ISODATA clustering on the selected images.

        If an ROI and prior labels are present, we currently run full-image ISODATA
        but inform the user that ROI-limited ISODATA is coming; KMeans already supports
        ROI-limited updates via the existing path.

        Returns:
            None
        """
        # Get user input for parameters
        k, ok = QInputDialog.getInt(self, "ISODATA Clustering", "Initial number of clusters:", 8, 1, 256)
        if not ok:
            return
        # Inform about ROI behavior for ISODATA (temporary)
        try:
            if self._spatial_roi is not None and self.labels is not None:
                QMessageBox.information(self, "ROI note", "ROI-limited ISODATA will be supported soon. Running full-image ISODATA for now.")
        except Exception:
            pass

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

        # ISODATA currently runs full-image; mark as non-ROI run
        self._last_run_was_roi = False

        # Start background worker with progress dialog (determinate for ISODATA)
        self._start_clustering_worker(
            algorithm='isodata',
            data=pixel_values,
            settings=settings,
            image_shape=(height, width),
            dialog_title=f"Running ISODATA ({max_iter} iterations max)...",
            determinate=True,
            maximum=100
        )

    def _start_clustering_worker(self, *, algorithm: str, data: np.ndarray, settings: typing.Any, image_shape: typing.Tuple[int, int], dialog_title: str, determinate: bool, maximum: int):
        # Create and show progress dialog
        self._progress_dialog = QProgressDialog("Preparing...", "Cancel", 0, maximum if determinate else 0, self)
        self._progress_dialog.setWindowTitle(dialog_title)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.setMinimumDuration(0)
        if not determinate:
            # Indeterminate/busy state
            self._progress_dialog.setRange(0, 0)
        else:
            self._progress_dialog.setValue(0)

        # Spin up thread and worker
        self._worker_thread = QThread(self)
        self._worker = ClusteringWorker(algorithm=algorithm, data=data, settings=settings, image_shape=image_shape, isodata_fn=self._isodata_algorithm)
        self._worker.moveToThread(self._worker_thread)

        # Wire signals
        self._worker_thread.started.connect(self._worker.run)

        def on_progress(pct: int, msg: str):
            try:
                if determinate and self._progress_dialog.maximum() != 0:
                    self._progress_dialog.setValue(max(0, min(100, pct)))
                self._progress_dialog.setLabelText(msg)
            except Exception:
                pass

        def cleanup_thread():
            try:
                self._worker_thread.quit()
                self._worker_thread.wait()
            except Exception:
                pass
            try:
                self._worker.deleteLater()
                self._worker_thread.deleteLater()
            except Exception:
                pass
            self._worker = None
            self._worker_thread = None

        def on_finished(labels, returned_settings):
            try:
                # Update state and invoke callback
                self.labels = labels
                n_clusters = len(np.unique(self.labels)) if isinstance(returned_settings, ISODATASettings) else returned_settings.n_clusters
                self.masks = self.generate_masks(self.labels, n_clusters)
                # Save to undo stack
                self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
                if len(self.undo_stack) > self.undo_stack_max_size:
                    self.undo_stack.pop(0)
                self.on_cluster_callback(self.labels, returned_settings)
            finally:
                try:
                    self._progress_dialog.reset()
                    self._progress_dialog.hide()
                except Exception:
                    pass
                cleanup_thread()

        def on_canceled():
            try:
                QMessageBox.information(self, "Clustering Canceled", "The clustering operation was canceled.")
            finally:
                try:
                    self._progress_dialog.reset()
                    self._progress_dialog.hide()
                except Exception:
                    pass
                cleanup_thread()

        def on_error(message: str):
            try:
                QMessageBox.critical(self, "Clustering Error", f"An error occurred during clustering:\n{message}")
            finally:
                try:
                    self._progress_dialog.reset()
                    self._progress_dialog.hide()
                except Exception:
                    pass
                cleanup_thread()

        self._worker.progress.connect(on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.canceled.connect(on_canceled)
        self._worker.error.connect(on_error)

        # Cancel button
        self._progress_dialog.canceled.connect(self._worker.request_cancel)

        # Start
        self._worker_thread.start()
        self._progress_dialog.show()

    def _isodata_algorithm(self, data: NDArray, settings: ISODATASettings, progress_cb: typing.Callable[[int, str], None] = None, cancel_event: threading.Event = None) -> NDArray[int]:
        """
        Implement the ISODATA clustering algorithm with optional progress and cancellation support.

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
            # Progress and cancel checks
            if cancel_event is not None and cancel_event.is_set():
                return None
            if progress_cb is not None:
                pct = int((iteration / max(1, settings.max_iter)) * 100)
                progress_cb(pct, f"ISODATA: Iteration {iteration+1}/{settings.max_iter}")
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

    def generate_distinct_colors(self, n_colors: int) -> List[QColor]:
        """
        Generate a list of perceptually distinct colors.

        This method creates a list of colors that are visually distinct from each other,
        suitable for visualizing different clusters. It uses a combination of predefined
        color palettes for small numbers of clusters and algorithmic generation for larger numbers.

        Args:
            n_colors (int): Number of distinct colors to generate.

        Returns:
            List[QColor]: List of QColor objects representing distinct colors.
        """
        # For small numbers of clusters, use a predefined set of distinct colors
        # These colors are chosen to be visually distinct and colorblind-friendly
        predefined_colors = [
            QColor(230, 25, 75),   # Red
            QColor(60, 180, 75),   # Green
            QColor(255, 225, 25),  # Yellow
            QColor(0, 130, 200),   # Blue
            QColor(245, 130, 48),  # Orange
            QColor(145, 30, 180),  # Purple
            QColor(70, 240, 240),  # Cyan
            QColor(240, 50, 230),  # Magenta
            QColor(210, 245, 60),  # Lime
            QColor(250, 190, 212), # Pink
            QColor(0, 128, 128),   # Teal
            QColor(220, 190, 255), # Lavender
            QColor(170, 110, 40),  # Brown
            QColor(255, 250, 200), # Beige
            QColor(128, 0, 0),     # Maroon
            QColor(170, 255, 195), # Mint
            QColor(128, 128, 0),   # Olive
            QColor(255, 215, 180), # Coral
            QColor(0, 0, 128),     # Navy
            QColor(128, 128, 128), # Grey
        ]

        if n_colors <= len(predefined_colors):
            return predefined_colors[:n_colors]

        # For larger numbers, use HSV color space with golden ratio to distribute hues
        colors = predefined_colors.copy()

        # Add more colors using the golden ratio method for hue distribution
        golden_ratio_conjugate = 0.618033988749895  # 1 / phi
        h = 0.1  # Starting hue
        s = 0.8  # Saturation
        v = 0.95  # Value

        while len(colors) < n_colors:
            h = (h + golden_ratio_conjugate) % 1.0
            # Vary saturation and value slightly for better distinction
            s_variation = 0.7 + (len(colors) % 3) * 0.1
            v_variation = 0.85 + (len(colors) % 2) * 0.1

            # Convert to RGB and create QColor
            h_degrees = h * 360.0
            color = QColor.fromHsv(int(h_degrees), int(s_variation * 255), int(v_variation * 255))
            colors.append(color)

        return colors

    def generate_masks(self, labels: NDArray[int], n_clusters: int) -> Dict[int, Tuple[NDArray[bool], QColor]]:
        """
        Generate binary masks and assign a unique color for each cluster.

        This method creates a binary mask for each unique cluster label and assigns
        a visually distinct color to each cluster using a perceptually-based approach.

        Args:
            labels (NDArray[int]): Array of cluster labels for each pixel.
            n_clusters (int): Number of clusters (may be different from actual unique labels).

        Returns:
            Dict[int, Tuple[NDArray[bool], QColor]]: Dictionary mapping cluster IDs to
                                                    tuples of (binary mask, color).
        """
        masks = {}
        unique_labels = np.unique(labels)

        # Generate distinct colors for all unique labels
        colors = self.generate_distinct_colors(len(unique_labels))

        for idx, cluster_id in enumerate(unique_labels):
            # Create binary mask for this cluster (True where label matches cluster_id)
            mask = (labels == cluster_id)

            # Assign a distinct color from our generated palette
            color = colors[idx]

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

    def isodata_on_mask(self, mask: NDArray[bool], settings: ISODATASettings) -> Tuple[NDArray[int], ISODATASettings]:
        """
        Run ISODATA clustering on pixels within the given mask using averaged image data.

        This method performs ISODATA clustering on a specific region of the image defined
        by the mask. It averages pixel values across selected images within the masked region,
        runs ISODATA on these averaged values, and updates the labels and masks accordingly.

        Args:
            mask (NDArray[bool]): Boolean mask indicating which pixels to include in clustering.
            settings (ISODATASettings): Settings for the ISODATA algorithm.

        Returns:
            Tuple[NDArray[int], ISODATASettings]: Tuple containing the new labels array and
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

        # Reshape for ISODATA algorithm: (n_features, n_samples) format
        # ISODATA expects features as rows, samples as columns
        avg_masked_pixels = avg_masked_pixels.reshape(1, -1)

        # Debug: Verify sizes
        print(f"Number of masked pixels: {avg_masked_pixels.shape[1]}")
        print(f"Number of True values in mask: {np.sum(mask)}")

        # Run ISODATA on averaged masked pixels using the algorithm method
        sub_labels = self._isodata_algorithm(avg_masked_pixels, settings)

        if sub_labels is None:
            return None, None

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
        self.undo_stack.append(
            (np.copy(self.labels), {k: (mask_data.copy(), color) for k, (mask_data, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        return new_labels, settings

    def merge_clusters(self, cluster_ids: List[int]) -> Tuple[NDArray[int], KMeansSettings]:
        """
        Merge multiple clusters into a single cluster.

        This method creates a new mask that combines the masks of all selected clusters,
        assigns a new label to all pixels in the combined mask, and updates the labels
        and masks accordingly.

        Args:
            cluster_ids (List[int]): List of cluster IDs to merge.

        Returns:
            Tuple[NDArray[int], KMeansSettings]: Tuple containing the new labels array and
                                               the settings used for clustering. Returns
                                               (None, None) if merging cannot be performed.
        """
        # Check if there are clusters to merge
        if not cluster_ids or len(cluster_ids) < 2:
            return None, None

        # Check if all cluster IDs are valid
        for cluster_id in cluster_ids:
            if cluster_id not in self.masks:
                return None, None

        # Create a combined mask for all selected clusters
        combined_mask = np.zeros_like(self.labels, dtype=bool)
        for cluster_id in cluster_ids:
            mask, _ = self.masks[cluster_id]
            combined_mask = np.logical_or(combined_mask, mask)

        # Create new labels array, preserving original labels outside the combined mask
        new_labels = np.copy(self.labels)

        # Get the maximum label value to ensure we use a new unique label
        max_label = np.max(self.labels) if self.labels is not None else -1
        new_cluster_id = max_label + 1

        # Assign the new label to all pixels in the combined mask
        new_labels[combined_mask] = new_cluster_id

        # Update labels and masks
        self.labels = new_labels
        self.masks = self.generate_masks(self.labels, len(np.unique(new_labels)))

        # Save state to undo stack
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        # Create settings for the callback
        settings = KMeansSettings(
            n_clusters=len(np.unique(self.labels)), 
            init="random", 
            n_init=5, 
            max_iter=100, 
            tol=1e-3, 
            random_state=42
        )

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
        # Note: QImage.Format_ARGB32 expects BGRA byte order in memory on little-endian systems.
        mask_data = np.zeros((height, width, 4), dtype=np.uint8)
        mask_data[mask_small, 0] = color.blue()   # B
        mask_data[mask_small, 1] = color.green()  # G
        mask_data[mask_small, 2] = color.red()    # R
        mask_data[mask_small, 3] = opacity        # A

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

        This method generates a list of perceptually distinct colors for visualizing
        cluster labels. It uses the same color generation approach as generate_masks
        to ensure consistency across different visualization methods.

        Args:
            num_labels (int): Number of unique labels

        Returns:
            List[int]: List of RGB values as integers
        """
        # Generate distinct colors using our perceptual color generation method
        colors = self.generate_distinct_colors(num_labels)

        # Convert QColors to qRgb integers
        return [qRgb(color.red(), color.green(), color.blue()) for color in colors]

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
