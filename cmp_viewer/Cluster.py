import typing
import cv2
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

class KMeansSettings(typing.NamedTuple):
    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    tol: float
    random_state: int

class ISODATASettings(typing.NamedTuple):
    max_clusters: int
    min_cluster_size: int
    max_iter: int
    tol: float
    split_threshold: float
    merge_threshold: float
    random_state: int

class Cluster(QWidget):
    def __init__(self, clusterImgName, clusterImages: cmp_viewer.models.ImageSet, selected_mask: NDArray[bool],
                 on_cluster_callback: Callable[[NDArray[int], Any], Tuple[NDArray[int], Any]]):
        super().__init__()
        self.on_cluster_callback = on_cluster_callback
        self.clusterImgName = clusterImgName
        self._image_set = clusterImages
        self._mask = selected_mask
        self.labels = None
        self.masks = None
        self.undo_stack = []
        self.undo_stack_max_size = 10

        layout = QVBoxLayout()
        self.clusterList = QListWidget()
        self.clusterList.addItems(["k-means", "ISODATA"])
        self.setWindowFlags(Qt.Dialog | Qt.Tool)

        widget = QWidget()
        self.button1 = QPushButton(widget)
        self.button1.setText("Run Clustering")
        self.button1.clicked.connect(self.run_clustering)

        layout.addWidget(self.clusterList)
        layout.addWidget(self.button1)
        self.setLayout(layout)

    def run_clustering(self):
        selected_method = self.clusterList.currentItem().text()
        if selected_method == "k-means":
            k, ok = QInputDialog.getInt(self, "K-Means Clustering", "Enter the value of k:", 8, 1, 256)
            if not ok:
                return
            self.runKMeansClustering(k)
        elif selected_method == "ISODATA":
            max_clusters, ok1 = QInputDialog.getInt(self, "ISODATA Clustering", "Maximum number of clusters:", 10, 2, 256)
            if not ok1:
                return
            min_cluster_size, ok2 = QInputDialog.getInt(self, "ISODATA Clustering", "Minimum cluster size:", 10, 1, 1000)
            if not ok2:
                return
            self.runISODATAClustering(max_clusters, min_cluster_size)

    def runKMeansClustering(self, k):
        pixels = self._image_set.images[self._mask, :, :].reshape(self._mask.sum(), -1)
        if pixels.size == 0:
            return
        settings = KMeansSettings(n_clusters=k, init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42)
        kmeans = KMeans(n_clusters=settings.n_clusters,
                        init=settings.init,
                        n_init=settings.n_init,
                        max_iter=settings.max_iter,
                        tol=settings.tol,
                        random_state=settings.random_state)
        kmeans.fit(pixels.T)
        self.labels = kmeans.labels_.reshape(self._image_set.image_shape)
        self.masks = self.generate_masks(self.labels, settings.n_clusters)
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)
        self.on_cluster_callback(self.labels, settings)

    def runISODATAClustering(self, max_clusters, min_cluster_size):
        # Prepare pixel data from the masked region
        if self._image_set.images.size == 0 or not np.any(self._mask):
            return

        # Debug: Print shapes to diagnose the issue
        print(f"Image set shape: {self._image_set.images.shape}")
        print(f"Mask shape: {self._mask.shape}")
        print(f"Number of True values in mask: {np.sum(self._mask)}")

        # Correctly extract pixels from the masked region
        # self._mask should be a 2D boolean array of shape (height, width)
        # self._image_set.images is (num_images, height, width)
        # We need to apply the mask to each image and stack the results
        height, width = self._image_set.image_shape
        if self._mask.shape != (height, width):
            raise ValueError(f"Mask shape {self._mask.shape} does not match image dimensions {self._image_set.image_shape}")

        # Flatten the mask to get indices of pixels to cluster
        mask_flat = self._mask.flatten()
        num_pixels = np.sum(mask_flat)

        if num_pixels == 0:
            print("No pixels selected in mask for clustering.")
            return

        # Extract pixel values for all images at the masked positions
        pixels = self._image_set.images[:, self._mask].T  # Shape: (num_pixels, num_images)
        print(f"Pixels shape for clustering: {pixels.shape}")

        if pixels.size == 0:
            print("No pixel data available after masking.")
            return

        settings = ISODATASettings(max_clusters=max_clusters, min_cluster_size=min_cluster_size,
                                  max_iter=100, tol=1e-3, split_threshold=1.0, merge_threshold=0.5,
                                  random_state=42)
        # Initial clustering with a small number of clusters
        k = min(2, max_clusters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, settings.max_iter, settings.tol)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        print(f"Labels shape after initial kmeans: {labels.shape}")

        # Labels should have shape (num_pixels, 1)
        labels = labels.flatten()
        print(f"Labels size after flatten: {labels.size}")

        if labels.size != num_pixels:
            raise ValueError(f"Labels size {labels.size} does not match number of pixels {num_pixels}")

        # Create a full label array with -1 for unmasked pixels
        full_labels = np.full((height * width,), -1, dtype=np.int32)
        full_labels[mask_flat] = labels
        self.labels = full_labels.reshape(self._image_set.image_shape)

        # Iterative splitting and merging (simplified ISODATA)
        for _ in range(settings.max_iter):
            unique_labels = np.unique(self.labels[self.labels >= 0])
            if len(unique_labels) >= settings.max_clusters:
                break
            # Split clusters with high variance
            new_labels = self.labels.copy()
            for label in unique_labels:
                mask = (self.labels == label)
                cluster_pixels = pixels[mask_flat[mask.flatten()]]
                if len(cluster_pixels) < settings.min_cluster_size:
                    continue
                variance = np.var(cluster_pixels, axis=0)
                if np.max(variance) > settings.split_threshold:
                    mean = np.mean(cluster_pixels, axis=0)
                    new_center = mean + 0.5 * np.std(cluster_pixels, axis=0)
                    centers = np.vstack([centers, new_center])
                    k += 1
                    if k > settings.max_clusters:
                        break
                    _, temp_labels, _ = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    temp_labels = temp_labels.flatten()
                    full_labels = np.full((height * width,), -1, dtype=np.int32)
                    full_labels[mask_flat] = temp_labels
                    new_labels = full_labels.reshape(self._image_set.image_shape)
            self.labels = new_labels

            # Merge clusters that are too close
            unique_labels = np.unique(self.labels[self.labels >= 0])
            if len(unique_labels) <= 2:
                break
            centers = np.array([np.mean(pixels[self.labels.flatten()[mask_flat] == label], axis=0) for label in unique_labels])
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < settings.merge_threshold:
                        mask_i = (self.labels == unique_labels[i])
                        mask_j = (self.labels == unique_labels[j])
                        self.labels[np.logical_or(mask_i, mask_j)] = unique_labels[i]
                        centers = np.delete(centers, j, axis=0)
                        k -= 1
                        break
                else:
                    continue
                break

        self.masks = self.generate_masks(self.labels, len(np.unique(self.labels[self.labels >= 0])))
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)
        self.on_cluster_callback(self.labels, settings)

    def generate_masks(self, labels: NDArray[int], n_clusters: int) -> Dict[int, Tuple[NDArray[bool], QColor]]:
        masks = {}
        unique_labels = np.unique(labels[labels >= 0])
        for idx, cluster_id in enumerate(unique_labels):
            hue = (idx * (360 / max(n_clusters, 1))) % 360
            lightness = 128 + (64 if idx % 2 == 0 else -64)
            color = QColor.fromHsl(int(hue), 255, lightness)
            mask = (labels == cluster_id)
            masks[cluster_id] = (mask, color)
        used_colors = [masks[cluster_id][1] for cluster_id in unique_labels]
        for i, cluster_id in enumerate(unique_labels):
            color = masks[cluster_id][1]
            for j, other_color in enumerate(used_colors[:i]):
                if i != j:
                    rgb_dist = ((color.red() - other_color.red())**2 +
                                (color.green() - other_color.green())**2 +
                                (color.blue() - other_color.blue())**2)**0.5
                    if rgb_dist < 50:
                        hue = (hue + 30) % 360
                        color = QColor.fromHsl(int(hue), 255, lightness)
                        masks[cluster_id] = (masks[cluster_id][0], color)
                        used_colors[i] = color
        return masks

    def cluster_on_mask(self, mask: NDArray[bool], n_clusters: int) -> Tuple[NDArray[int], KMeansSettings]:
        if self._image_set.images.size == 0 or not np.any(mask):
            return None, None
        selected_images = self._image_set.images[self._mask, :, :]
        if selected_images.size == 0:
            return None, None
        if mask.shape != selected_images.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} does not match image dimensions {selected_images.shape[1:]}")
        masked_images = [img[mask] for img in selected_images]
        if not masked_images or all(len(m) == 0 for m in masked_images):
            return None, None
        avg_masked_pixels = np.mean(np.vstack(masked_images), axis=0)
        avg_masked_pixels = avg_masked_pixels.reshape(-1, 1)
        print(f"Number of masked pixels: {avg_masked_pixels.shape[0]}")
        print(f"Number of True values in mask: {np.sum(mask)}")
        settings = KMeansSettings(n_clusters=n_clusters, init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42)
        kmeans = KMeans(n_clusters=settings.n_clusters,
                        init=settings.init,
                        n_init=settings.n_init,
                        max_iter=settings.max_iter,
                        tol=settings.tol,
                        random_state=settings.random_state)
        sub_labels = kmeans.fit_predict(avg_masked_pixels)
        print(f"Sub_labels size: {sub_labels.shape[0]}")
        new_labels = np.copy(self.labels)
        max_label = np.max(self.labels) if self.labels is not None else -1
        new_labels[mask] = sub_labels + max_label + 1
        self.labels = new_labels
        self.masks = self.generate_masks(self.labels, len(np.unique(new_labels)))
        self.undo_stack.append((np.copy(self.labels), {k: (mask.copy(), color) for k, (mask, color) in self.masks.items()}))
        if len(self.undo_stack) > self.undo_stack_max_size:
            self.undo_stack.pop(0)
        return new_labels, settings

    def undo_clustering(self):
        if len(self.undo_stack) <= 1:
            return False
        self.undo_stack.pop()
        prev_labels, prev_masks = self.undo_stack[-1]
        self.labels = np.copy(prev_labels)
        self.masks = prev_masks
        self.on_cluster_callback(self.labels, KMeansSettings(n_clusters=len(np.unique(self.labels)), init="random", n_init=5, max_iter=100, tol=1e-3, random_state=42))
        return True
