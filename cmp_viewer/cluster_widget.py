import typing
from typing import Callable, Tuple, List, Any, Dict
import os
import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtGui import QImage, qRgb, QColor
from sklearn.cluster import KMeans
from PIL import Image
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QInputDialog, QProgressDialog, QMessageBox
import cmp_viewer.models
import cmp_viewer.utils
import logging
from cmp_viewer.clustering_algorithms import ClusteringWorker, isodata_algorithm
from cmp_viewer.utils import KMeansSettings, ISODATASettings
import cmp_viewer.mask as mask_module
from cmp_viewer import utils

# Create Logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'Logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging to write to BOTH file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'cluster_debug.log')),  # Log to file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("=== cluster_widget.py module loaded ===")


"""
This module provides clustering functionality for multidimensional images.
It implements K-means clustering, ISODATA clustering, and visualization of clustered images.
"""


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
                 *, base_labels: NDArray[int] | None = None, base_masks: dict | None = None,
                 spatial_roi: NDArray[bool] | None = None):
        """
        Initialize the Cluster widget.

        Args:
            clusterImgName: Name of the clustered image.
            clusterImages (ImageSet): Set of images to cluster.
            selected_mask (NDArray[bool]): Boolean mask indicating which images to include.
            on_cluster_callback (Callable): Function to call after clustering is complete.
            base_labels: Existing cluster labels (for ROI-based clustering).
            base_masks: Existing cluster masks (for ROI-based clustering).
            spatial_roi: Spatial ROI mask for constrained clustering.
        """
        super().__init__()
        logger.info(f"=== Cluster.__init__ called ===")
        logger.info(f"spatial_roi parameter: {spatial_roi is not None}")
        if spatial_roi is not None:
            logger.info(
                f"spatial_roi shape: {spatial_roi.shape}, dtype: {spatial_roi.dtype}, True pixels: {np.sum(spatial_roi)}")
        logger.info(f"base_labels: {base_labels is not None}")
        logger.info(f"base_masks: {base_masks is not None}")

        self.on_cluster_callback = on_cluster_callback
        self.clusterImgName = clusterImgName
        self._image_set = clusterImages
        self._mask = selected_mask
        # Prior state and ROI
        self.labels = base_labels if base_labels is not None else None  # Current cluster labels
        self.masks = base_masks if base_masks is not None else None  # Dict[int, Tuple[NDArray[bool], QColor]]
        self._spatial_roi = spatial_roi if spatial_roi is not None else None  # Optional ROI mask

        logger.info(f"After assignment - self._spatial_roi: {self._spatial_roi is not None}")
        if self._spatial_roi is not None:
            logger.info(f"self._spatial_roi shape: {self._spatial_roi.shape}, True pixels: {np.sum(self._spatial_roi)}")

        self.undo_stack = []  # Stack to store previous states (labels, masks)
        self.undo_stack_max_size = 10  # Limit undo stack size
        # Track whether the last run used an ROI (for viewer naming/clearing logic)
        self._last_run_was_roi = False
        # If prior state exists, seed undo stack
        if self.labels is not None and self.masks is not None:
            try:
                self.undo_stack.append(
                    (np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
                logger.info("Seeded undo stack with prior state")
            except Exception as e:
                logger.warning(f"Failed to seed undo stack: {e}")

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

        logger.info("=== Cluster.__init__ completed ===")

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
            logger.info(f"Unknown clustering algorithm: {selected_algorithm}")

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
        logger.info("=== Starting ISODATA clustering ===")
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

        logger.info(f"ISODATA parameters: max_iter={max_iter}, min_samples={min_samples}, max_std_dev={max_std_dev}, min_cluster_distance={min_cluster_distance}, max_merge_pairs={max_merge_pairs}")

        # Path 1: ROI-based re-clustering if prior labels and ROI are available
        if self._spatial_roi is not None and self.labels is not None:
            logger.info("ROI and prior labels detected - attempting ROI-based ISODATA clustering")
            try:
                settings = ISODATASettings(
                    n_clusters=k,
                    max_iter=max_iter,
                    min_samples=min_samples,
                    max_std_dev=max_std_dev,
                    min_cluster_distance=min_cluster_distance,
                    max_merge_pairs=max_merge_pairs,
                    random_state=42
                )
                logger.info(
                    f"ROI mask has {np.sum(self._spatial_roi)} True pixels out of {self._spatial_roi.size} total pixels")

                new_labels, returned_settings = self.isodata_on_mask(self._spatial_roi, settings)
                if new_labels is None:
                    logger.warning("ROI clustering returned None - ROI may be empty")
                    QMessageBox.warning(self, "ROI Empty",
                                        "No pixels in the selected ROI. Please choose a different mask or run full-image clustering.")
                    return

                # Mark this run as ROI-based
                self._last_run_was_roi = True
                logger.info("ROI-based ISODATA completed successfully")
                # Callback will update masks and main view
                self.on_cluster_callback(new_labels, returned_settings)
                return
            except Exception as e:
                logger.error(f"ROI clustering failed with error: {e}", exc_info=True)
                QMessageBox.warning(self, "ROI Clustering Error",
                                    f"Falling back to full-image ISODATA due to error: {e}")
                # fall-through to full-image path

        # Path 2: Full-image ISODATA
        logger.info("Running full-image ISODATA (no ROI)")
        # Get the image shape for later reshaping
        height, width = self._image_set.image_shape
        logger.info(f"Image shape: {height}x{width}")

        # Get the pixel values for all selected images
        pixel_values = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)
        logger.info(f"Processing {self._mask.sum()} images, total pixels: {pixel_values.size}")

        if pixel_values.size == 0:
            logger.warning("No pixel data to cluster")
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

        # ISODATA full-image; mark as non-ROI run
        self._last_run_was_roi = False
        logger.info("Starting full-image ISODATA worker thread")

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
        self._worker = ClusteringWorker(algorithm=algorithm, data=data, settings=settings, image_shape=image_shape, isodata_fn=isodata_algorithm)
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
                self.masks = mask_module.generate_masks(self.labels, n_clusters)
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
        logger.info(f"Number of masked pixels: {avg_masked_pixels.shape[0]}")
        logger.info(f"Number of True values in mask: {np.sum(mask)}")

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
        logger.info(f"Sub_labels size: {sub_labels.shape[0]}")

        # Create new labels array, preserving original labels outside the mask
        new_labels = np.copy(self.labels)
        max_label = np.max(self.labels) if self.labels is not None else -1
        new_labels[mask] = sub_labels + max_label + 1  # Offset new labels to avoid overlap

        # Update masks with the new labels
        self.labels = new_labels
        self.masks = mask_module.generate_masks(self.labels, len(np.unique(new_labels)))

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
        logger.info("=== isodata_on_mask called ===")

        # Check if there are images to cluster or if the mask is empty
        if self._image_set.images.size == 0 or not np.any(mask):
            logger.warning(
                f"Cannot cluster: images.size={self._image_set.images.size}, mask has {np.sum(mask)} True values")
            return None, None

        # Get the selected images
        selected_images = self._image_set.images[self._mask, :, :]
        logger.info(f"Selected {selected_images.shape[0]} images with shape {selected_images.shape[1:]} each")

        if selected_images.size == 0:
            logger.warning("No selected images")
            return None, None

        # Ensure mask matches the image dimensions (height, width)
        if mask.shape != selected_images.shape[1:]:
            error_msg = f"Mask shape {mask.shape} does not match image dimensions {selected_images.shape[1:]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Average pixel values across selected images within the mask
        masked_images = [img[mask] for img in selected_images]
        logger.info(f"Extracted masked pixels from {len(masked_images)} images")

        if not masked_images or all(len(m) == 0 for m in masked_images):
            logger.warning("All masked images are empty")
            return None, None

        avg_masked_pixels = np.mean(np.vstack(masked_images), axis=0)

        # Reshape for ISODATA algorithm: (n_features, n_samples) format
        avg_masked_pixels = avg_masked_pixels.reshape(1, -1)

        logger.info(f"Number of masked pixels: {avg_masked_pixels.shape[1]}")
        logger.info(f"Number of True values in mask: {np.sum(mask)}")

        # Run ISODATA on averaged masked pixels using the algorithm method
        logger.info(f"Starting ISODATA algorithm with {settings.n_clusters} initial clusters")
        sub_labels = isodata_algorithm(avg_masked_pixels, settings)

        if sub_labels is None:
            logger.warning("ISODATA algorithm returned None")
            return None, None

        logger.info(f"ISODATA completed. Sub_labels size: {sub_labels.shape[0]}")
        logger.info(f"Unique labels in sub_labels: {np.unique(sub_labels)}")

        # Create new labels array, preserving original labels outside the mask
        new_labels = np.copy(self.labels)
        max_label = np.max(self.labels) if self.labels is not None else -1
        logger.info(f"Max label in existing labels: {max_label}")

        new_labels[mask] = sub_labels + max_label + 1  # Offset new labels to avoid overlap
        logger.info(f"Assigned new labels with offset {max_label + 1}")

        # Update masks with the new labels
        self.labels = new_labels
        self.masks = mask_module.generate_masks(self.labels, len(np.unique(new_labels)))
        logger.info(f"Generated {len(self.masks)} masks")

        # Save state to undo stack
        self.undo_stack.append(
            (np.copy(self.labels), {k: (mask_data.copy(), color) for k, (mask_data, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)

        logger.info("=== isodata_on_mask completed successfully ===")
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
        self.masks = mask_module.generate_masks(self.labels, len(np.unique(new_labels)))

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












