# CMP Viewer

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

- numpy
- PyQt5
- scikit-learn
- Pillow
- opencv-python (for mac install use opencv-python-headless)
- nornir_imageregistration
- qimage2ndarray

## License

This project is licensed under the MIT License - see the LICENSE file for details.
