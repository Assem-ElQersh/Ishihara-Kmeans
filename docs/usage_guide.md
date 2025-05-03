# Ishihara K-means Usage Guide

This guide explains how to use the Ishihara K-means package to extract numbers from Ishihara color blindness test images.

## Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/ishihara-kmeans.git
cd ishihara-kmeans

# Install the package
pip install -e .
```

### Dependencies

The package requires the following dependencies:
- Python 3.7+
- NumPy
- OpenCV (cv2)
- Matplotlib

These will be installed automatically when you install the package.

## Basic Usage

### As a Command-Line Tool

The simplest way to use the package is through the command-line interface:

```bash
# Basic usage with default parameters (Lab color space, a-channel, k=2)
ishihara-kmeans data/ishihara_74.png

# Specify a different color space and parameters
ishihara-kmeans data/ishihara_74.png --color-space lab --channel 1 --k 2

# Invert the result (sometimes needed for certain color spaces)
ishihara-kmeans data/ishihara_74.png --color-space rgb --channel 1 --k 2 --invert

# Find the optimal parameters automatically
ishihara-kmeans data/ishihara_74.png --optimize

# Get recommendations for the best color space and channel
ishihara-kmeans data/ishihara_74.png --recommend

# Save results to a specific directory
ishihara-kmeans data/ishihara_74.png --output results/

# Don't display results, just save them
ishihara-kmeans data/ishihara_74.png --no-display
```

### As a Python Package

You can also use the package in your own Python code:

```python
from ishihara_kmeans import KMeansClustering, preprocess_image, segment_image, extract_number

# Load and preprocess the image
image_rgb, features, _ = preprocess_image("data/ishihara_74.png", 'lab', 1)

# Apply K-means clustering
kmeans = KMeansClustering(k=2, max_iterations=100)
kmeans.fit(features)

# Segment the image
labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, 1)

# Extract the number
number_mask = extract_number(labels, 2)

# Display or save the results
# ...
```

## Advanced Usage

### Color Space Selection

The package supports multiple color spaces for improved number extraction:

```python
# RGB color space
image_rgb, features, _ = preprocess_image("image.png", 'rgb', 0)  # Red channel

# HSV color space
image_rgb, features, _ = preprocess_image("image.png", 'hsv', 0)  # Hue channel

# Lab color space
image_rgb, features, _ = preprocess_image("image.png", 'lab', 1)  # a channel (red-green)

# YCrCb color space
image_rgb, features, _ = preprocess_image("image.png", 'ycrcb', 1)  # Cr channel (red-difference)
```

### Automatic Optimization

To find the optimal parameters automatically:

```python
from ishihara_kmeans import analyze_with_multiple_params

# Analyze with multiple parameter combinations
original_image, results = analyze_with_multiple_params("image.png")

# Get the best parameters
best_params = results[0][0]
print(f"Best parameters: {best_params['color_space']}, channel {best_params['channel']}, k={best_params['k']}")
```

### Color Space Recommendations

To get recommendations for the best color space and channel:

```python
import cv2
from ishihara_kmeans import recommend_color_space

# Read the image
image = cv2.imread("image.png")

# Get recommendations
recommendations = recommend_color_space(image)

# Use the top recommendation
top_rec = recommendations[0]
print(f"Recommended: {top_rec['color_space']}, channel {top_rec['channel']}")
```

### Visualization

The package provides several visualization functions:

```python
from ishihara_kmeans import plot_results, create_final_visualization

# Plot the original image, segmented image, and extracted number
plot_results(image_rgb, segmented_image, number_mask, "Lab color space, a channel")

# Create a final visualization with enhanced number visibility
enhanced_number = create_final_visualization(image_rgb, number_mask, "result.png")
```

### Analyzing Color Channels

To visualize different color channels:

```python
from ishihara_kmeans import visualize_color_channels

# Visualize RGB channels
fig, _, _, _ = visualize_color_channels(image, 'rgb')

# Visualize Lab channels
fig, _, _, _ = visualize_color_channels(image, 'lab')
```

## Best Practices

### For Standard Ishihara Tests

For most standard Ishihara color blindness tests:

1. Try the Lab color space, 'a' channel, k=2 first
2. If that doesn't work well, try YCrCb, 'Cr' channel, k=2
3. For more complex patterns, try HSV color space, 'H' channel, k=3

### Parameter Tuning

If you need to tune the parameters:

- **Number of clusters (k)**: Start with k=2, increase if needed
- **Inversion**: Some color spaces might need inversion to properly show the number
- **Post-processing**: Adjust the kernel size in `extract_number()` for better noise removal

## Example Workflow

Here's a complete example workflow:

```python
import cv2
import matplotlib.pyplot as plt
from ishihara_kmeans import (
    KMeansClustering, 
    preprocess_image, 
    segment_image, 
    extract_number,
    recommend_color_space,
    create_final_visualization
)

# Read the image
image = cv2.imread("ishihara_test.png")

# Get recommendations
recommendations = recommend_color_space(image)
color_space = recommendations[0]['color_space']
channel = recommendations[0]['channel']

# Preprocess image
image_rgb, features, _ = preprocess_image("ishihara_test.png", color_space, channel)

# Apply K-means clustering
kmeans = KMeansClustering(k=2, max_iterations=100)
kmeans.fit(features)

# Segment image
labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, channel)

# Extract number
number_mask = extract_number(labels, 2)

# Create final visualization
create_final_visualization(image_rgb, number_mask, "result.png")
```

## Troubleshooting

### Number Not Visible

If the number is not visible in the extracted mask:

- Try a different color space (Lab, YCrCb, HSV, RGB)
- Try a different channel within the same color space
- Increase the number of clusters (k) to 3 or 4
- Try inverting the result with the `--invert` flag or `invert=True` parameter

### Poor Segmentation

If the segmentation is poor:

- Make sure the image is properly loaded (check for transparency, etc.)
- Try different post-processing parameters in `extract_number()`
- Use the `--optimize` flag to try multiple parameter combinations

### Errors

If you encounter errors:

- Ensure all dependencies are installed
- Check that the image path is correct and the image is readable
- Verify that the color space and channel parameters are valid

## Further Resources

For more information, check out the other documentation:

- [Color Spaces Guide](color_spaces.md)
- [K-means Algorithm Explanation](kmeans_algorithm.md)

Also, look at the examples in the `examples/` directory for more usage patterns.
