"""
Ishihara K-means: A package for extracting numbers from Ishihara color blindness test images.

This package implements K-means clustering from scratch and applies it to
extract hidden numbers from Ishihara color blindness test images.
"""

__version__ = '1.0.0'

from .kmeans import KMeansClustering
from .image_processing import (
    preprocess_image, 
    segment_image, 
    extract_number, 
    analyze_with_multiple_params
)
from .color_spaces import (
    convert_to_color_space,
    visualize_color_channels,
    analyze_channel_histograms,
    compare_color_spaces,
    recommend_color_space
)
from .visualization import (
    plot_results,
    plot_color_space_comparison,
    save_comparison_figure,
    create_visualization_grid,
    enhance_number_visibility,
    create_final_visualization
)
