# Ishihara Test Number Extraction using K-means Clustering

A Python library for extracting hidden numbers from Ishihara color blindness test images using custom K-means clustering implementation.

## Overview

This project implements the K-means clustering algorithm from scratch and applies it to extract numbers from Ishihara color blindness test images. The implementation leverages different color spaces (RGB, HSV, Lab, YCrCb) to optimize number extraction based on the specific properties of Ishihara tests.

![Example Result](docs/example_result.jpg)

## Features

- K-means clustering implementation from scratch
- Support for multiple color spaces (RGB, HSV, Lab, YCrCb)
- Channel-specific processing for targeted analysis
- Automatic parameter optimization
- Visualization utilities for results analysis
- Command-line interface for easy usage

## Installation

```bash
# Clone the repository
git clone https://github.com/Assem-ElQersh/Ishihara-Kmeans.git
cd Ishihara-Kmeans

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.kmeans import KMeansClustering
from src.image_processing import preprocess_image, segment_image, extract_number
import matplotlib.pyplot as plt

# Load and preprocess image using Lab color space, a-channel
image_rgb, features, _ = preprocess_image("data/74.jpg", 'lab', 1)

# Apply K-means clustering with 2 clusters
kmeans = KMeansClustering(k=2, max_iterations=100)
kmeans.fit(features)

# Segment the image and extract the number
labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, 1)
number_mask = extract_number(labels, 2)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(segmented_image, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(number_mask, cmap="gray")
plt.title("Extracted Number")
plt.axis("off")

plt.tight_layout()
plt.show()
```

### Command Line Usage

```bash
# Basic usage with default parameters (Lab color space, a-channel, k=2)
python -m src.cli data/74.jpg

# Try different color spaces and parameters
python -m src.cli data/74.jpg --color-space lab --channel 1 --k 2

# Optimize parameters automatically
python -m src.cli data/74.jpg --optimize
```

## Project Structure

- **src/**: Source code including K-means implementation and image processing utilities
- **examples/**: Example scripts and notebooks demonstrating usage
- **tests/**: Unit tests
- **data/**: Sample Ishihara test images
- **docs/**: Documentation and guides

## Color Spaces for Ishihara Tests

Different color spaces can significantly improve the accuracy of number extraction:

- **Lab**: The 'a' channel represents red-green differences, making it ideal for typical Ishihara tests
- **YCrCb**: The 'Cr' channel represents red-difference, also very effective
- **HSV**: The 'H' (hue) channel can be useful for tests with unique color patterns
- **RGB**: The red or green channels can sometimes work well for specific tests

See [Color Spaces Guide](docs/color_spaces.md) for more details.

## Documentation

- [Usage Guide](docs/usage_guide.md)
- [K-means Algorithm Explanation](docs/kmeans_algorithm.md)
- [Color Spaces Guide](docs/color_spaces.md)

## Requirements

- Python 3.7+
- NumPy
- OpenCV
- Matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
