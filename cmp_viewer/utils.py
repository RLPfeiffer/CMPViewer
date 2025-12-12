import typing
from typing import List, Tuple, NamedTuple
from PyQt5.QtGui import QColor, qRgb
from PIL import Image
import numpy as np
from numpy.typing import NDArray

"""
This module provides utility functions for image processing and conversion
between different image formats used in the CMP Viewer application.
"""

def numpy_labels_to_pillow_image(input: NDArray[int]) -> Image:
    """
    Convert a NumPy array of integer labels to a PIL Image in palette mode.

    This function is particularly useful for visualizing clustered images where
    each pixel value represents a cluster label. The resulting image uses the
    'P' (palette) mode, which is suitable for images with a limited number of colors.

    Args:
        input (NDArray[int]): A 2D NumPy array containing integer labels.
                             Shape should be (height, width).

    Returns:
        Image: A PIL Image object in palette mode ('P') with the same dimensions
              as the input array. Each pixel value corresponds to the label in
              the input array.

    Note:
        The palette of the output image is not set by this function. To visualize
        the labels with distinct colors, you may need to set a custom palette.
    """
    # Create a new image with the same size as the original image
    output_img = Image.new('P', (input.shape[1], input.shape[0]))
    output_img.putdata(np.array(input.flat))
    return output_img


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


def generate_distinct_colors(n_colors: int) -> List[QColor]:
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


def create_color_table(num_labels: int) -> List[int]:
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
    colors = generate_distinct_colors(num_labels)

    # Convert QColors to qRgb integers
    return [qRgb(color.red(), color.green(), color.blue()) for color in colors]


def create_palette_from_color_table(color_table: List[int]) -> List[int]:
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


def prepare_label_image_for_display(img: Image.Image, num_labels: int) -> Tuple[Image.Image, List[int]]:
    """
    Prepare a label image for display by setting its palette.

    Args:
        img (Image.Image): PIL Image with label data
        num_labels (int): Number of unique labels

    Returns:
        Tuple[Image.Image, List[int]]: Tuple containing the prepared image and color table
    """
    # Create color table and palette
    color_table = create_color_table(num_labels)
    palette = create_palette_from_color_table(color_table)

    # Convert image to palette mode if needed
    if img.mode != 'P':
        img = img.convert('P')

    # Set the palette
    img.putpalette(palette)

    return img, color_table


def calculate_optimal_scale_factor(height: int, width: int, max_pixels: int = 500000) -> float:
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