# CMP Viewer (v1.5.2)

A cross-platform tool for visualizing and analyzing multidimensional images with clustering capabilities, written in Python by Becca Pfeiffer and Nat Quayle Nelson.

## Description

CMP Viewer is a tool for visualizing and analyzing multidimensional images. It provides functionality for loading images, performing clustering operations, and visualizing the results. The application uses PyQt5 for its graphical user interface.

## Installation

### Using pip

```bash
pip install cmp_viewer
```

### From source

```bash
git clone https://github.com/yourusername/cmp_viewer.git
cd cmp_viewer
pip install -e .
```

## Usage

After installation, you can run the application using:

```bash
cmp-viewer
```

Or from Python:

```python
from cmp_viewer.__main__ import main
main()
```

## Dependencies

- numpy (>=2.2.5)
- PyQt5
- scikit-learn
- Pillow (>=11.2.1)
- opencv-python (for mac install use opencv-python-headless)
- nornir_imageregistration (from GitHub)
- nornir_shared (from GitHub)
- nornir_pools (from GitHub)
- qimage2ndarray
- matplotlib (>=3.10.3)
- scipy (>=1.15.3)
- scikit-image (>=0.25.2)
- imageio (>=2.37.0)
- tifffile (>=2025.5.10)
- networkx (>=3.4.2)
- pydantic (>=2.11.4)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
