"""
Clustering algorithm implementations (K-means, ISODATA).
"""
import typing
import threading
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.cluster import KMeans
import logging
from cmp_viewer.utils import ISODATASettings, KMeansSettings

logger = logging.getLogger(__name__)


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

def isodata_algorithm(data: NDArray, settings: ISODATASettings, progress_cb: typing.Callable[[int, str], None] = None, cancel_event: threading.Event = None) -> NDArray[int]:

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
    logger.info(f"=== _isodata_algorithm started ===")
    logger.info(f"Data shape: {data.shape} (features x samples)")
    logger.info(f"Initial clusters: {settings.n_clusters}")

    np.random.seed(settings.random_state)
    n_samples = data.shape[1]
    n_features = data.shape[0]

    # Adjust number of clusters if it exceeds number of samples
    settings = settings._replace(n_clusters=min(settings.n_clusters, n_samples))
    logger.info(f"Adjusted clusters to {settings.n_clusters} (cannot exceed {n_samples} samples)")

    # Initialize centroids randomly
    # Select k random samples as initial centroids
    indices = np.random.choice(n_samples, settings.n_clusters, replace=False)
    centroids = data[:, indices]
    logger.info(f"Initialized {settings.n_clusters} centroids")

    # Initialize labels
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(settings.max_iter):
        logger.info(f"--- Iteration {iteration + 1}/{settings.max_iter} ---")

        # Progress and cancel checks
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Algorithm canceled by user")
            return None
        if progress_cb is not None:
            pct = int((iteration / max(1, settings.max_iter)) * 100)
            progress_cb(pct, f"ISODATA: Iteration {iteration+1}/{settings.max_iter}")
        # Store current number of clusters before any modifications
        old_n_clusters = settings.n_clusters

        # Assign samples to closest centroids (like k-means)
        logger.debug(f"Assigning samples to {settings.n_clusters} centroids...")
        distances = np.zeros((settings.n_clusters, n_samples))
        for i in range(settings.n_clusters):
            diff = data - centroids[:, i].reshape(-1, 1)
            distances[i] = np.sum(diff**2, axis=0)

        # Assign each sample to the closest centroid
        labels = np.argmin(distances, axis=0)
        unique_labels_assigned = len(np.unique(labels))
        logger.info(f"Assigned samples to {unique_labels_assigned} unique clusters")

        # Make a copy of the current centroids for convergence check
        old_centroids = centroids.copy()

        # Update centroids based on new assignments
        logger.debug("Updating centroids...")
        for i in range(settings.n_clusters):
            cluster_samples = data[:, labels == i]
            if cluster_samples.shape[1] > 0:
                centroids[:, i] = np.mean(cluster_samples, axis=1)

        # Check for empty clusters and handle them
        empty_clusters = [i for i in range(settings.n_clusters) if np.sum(labels == i) == 0]
        if empty_clusters:
            logger.info(f"Handling {len(empty_clusters)} empty clusters")

        for i in range(settings.n_clusters):
            if np.sum(labels == i) == 0:
                # Find the cluster with the most samples
                largest_cluster = np.argmax([np.sum(labels == j) for j in range(settings.n_clusters)])
                # Find the samples furthest from the centroid in the largest cluster
                cluster_samples = data[:, labels == largest_cluster]
                if cluster_samples.shape[1] > 0:
                    diff = cluster_samples - centroids[:, largest_cluster].reshape(-1, 1)
                    distances = np.sum(diff ** 2, axis=0)
                    furthest_sample_idx = np.argmax(distances)
                    # Set the empty cluster's centroid to this sample
                    centroids[:, i] = cluster_samples[:, furthest_sample_idx]
                    # Reassign some samples to this new centroid
                    diff = data - centroids[:, i].reshape(-1, 1)
                    new_distances = np.sum(diff ** 2, axis=0)
                    closest_to_new = np.argsort(new_distances)[:settings.min_samples]
                    labels[closest_to_new] = i

        # ISODATA specific steps:
        logger.debug("Applying ISODATA operations...")

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

        # At the end of iteration
        logger.info(f"End of iteration {iteration + 1}: {settings.n_clusters} clusters")

        # Check for convergence only if the number of clusters hasn't changed
        if old_n_clusters == settings.n_clusters:
            if np.allclose(old_centroids[:, :settings.n_clusters], centroids):
                logger.info(f"Converged at iteration {iteration + 1}")
                break
        # If number of clusters changed, continue to next iteration
        else:
            logger.info(f"Number of clusters changed from {old_n_clusters} to {settings.n_clusters}")
            continue

    # Ensure labels are consecutive integers starting from 0
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.array([label_map[l] for l in labels])
    logger.info(f"=== ISODATA completed with {len(unique_labels)} final clusters ===")

    return new_labels