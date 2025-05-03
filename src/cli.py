#!/usr/bin/env python3
"""
Command-line interface for Ishihara K-means number extraction
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .kmeans import KMeansClustering
from .image_processing import preprocess_image, segment_image, extract_number, analyze_with_multiple_params
from .visualization import plot_results, plot_color_space_comparison, save_comparison_figure, create_final_visualization
from .color_spaces import recommend_color_space

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract numbers from Ishihara color blindness test images using K-means clustering"
    )
    parser.add_argument(
        "image_path", 
        help="Path to the input Ishihara test image"
    )
    parser.add_argument(
        "--color-space", 
        choices=["rgb", "hsv", "lab", "ycrcb"], 
        default="lab",
        help="Color space to use for clustering (default: lab)"
    )
    parser.add_argument(
        "--channel", 
        type=int, 
        help="Specific channel to use for clustering (default: 1 for lab)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=2,
        help="Number of clusters (default: 2)"
    )
    parser.add_argument(
        "--invert", 
        action="store_true",
        help="Invert the extracted number mask"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Try different color spaces and parameters to find the best combination"
    )
    parser.add_argument(
        "--recommend", 
        action="store_true",
        help="Recommend the best color space and channel for this image"
    )
    parser.add_argument(
        "--output", 
        help="Output directory for saving results (default: current directory)"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Don't display results (just save them)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function for command-line interface
    """
    # Parse arguments
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output if args.output else "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default channel based on color space if not specified
    if args.channel is None:
        if args.color_space == "lab":
            args.channel = 1  # a channel
        elif args.color_space == "ycrcb":
            args.channel = 1  # Cr channel
        elif args.color_space == "hsv":
            args.channel = 0  # Hue channel
    
    # Get base filename for output
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Handle recommendation mode
    if args.recommend:
        # Read image
        image = cv2.imread(args.image_path)
        
        # Get recommendations
        recommendations = recommend_color_space(image)
        
        print("\nRecommended color spaces and channels for this image:")
        print("------------------------------------------------------")
        for i, rec in enumerate(recommendations):
            print(f"Recommendation {i+1}:")
            print(f"  Color space: {rec['color_space'].upper()}")
            print(f"  Channel: {rec['channel']}")
            print(f"  Explanation: {rec['explanation']}")
            print("")
        
        # Use the top recommendation
        top_rec = recommendations[0]
        print(f"Using top recommendation: {top_rec['color_space'].upper()} color space, channel {top_rec['channel']}\n")
        
        # Update arguments
        args.color_space = top_rec['color_space']
        args.channel = top_rec['channel']
    
    # Handle optimization mode
    if args.optimize:
        print(f"Analyzing image with multiple parameter combinations: {args.image_path}")
        
        # Analyze with multiple parameters
        original_image, results = analyze_with_multiple_params(args.image_path)
        
        # Save comparison figure
        output_path = os.path.join(output_dir, f"{base_filename}_comparison.png")
        save_comparison_figure(original_image, results, output_path)
        
        # Display results if needed
        if not args.no_display:
            plot_color_space_comparison(original_image, results)
            plt.show()
        
        # Use the best result for final output
        best_params = results[0][0]
        print(f"\nBest parameters found:")
        print(f"  Color space: {best_params['color_space'].upper()}")
        print(f"  Channel: {best_params['channel']}")
        print(f"  k: {best_params['k']}")
        if best_params['invert']:
            print(f"  Invert: Yes")
        
        # Update arguments to use the best parameters for final visualization
        args.color_space = best_params['color_space']
        args.channel = best_params['channel']
        args.k = best_params['k']
        args.invert = best_params['invert']
    
    # Process with specific parameters (either provided or determined from optimization)
    print(f"Processing image: {args.image_path}")
    print(f"Parameters: {args.color_space.upper()} color space" + 
          (f", channel {args.channel}" if args.channel is not None else "") + 
          f", k={args.k}" + (", inverted" if args.invert else ""))
    
    # Preprocess image
    image_rgb, features, _ = preprocess_image(args.image_path, args.color_space, args.channel)
    
    # Apply K-means clustering
    print("Fitting K-means clustering...")
    kmeans = KMeansClustering(k=args.k, max_iterations=100)
    kmeans.fit(features)
    
    # Segment image
    print("Segmenting image...")
    labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, args.channel)
    
    # Extract number
    print("Extracting number...")
    number_mask = extract_number(labels, args.k, args.invert)
    
    # Save results
    segmented_path = os.path.join(output_dir, f"{base_filename}_segmented.png")
    number_path = os.path.join(output_dir, f"{base_filename}_number.png")
    
    if args.channel is not None and len(segmented_image.shape) == 2:
        plt.imsave(segmented_path, segmented_image, cmap="gray")
    else:
        plt.imsave(segmented_path, segmented_image)
    
    plt.imsave(number_path, number_mask, cmap="gray")
    
    # Create final visualization
    final_path = os.path.join(output_dir, f"{base_filename}_result.png")
    create_final_visualization(image_rgb, number_mask, final_path)
    
    # Display results if needed
    if not args.no_display:
        title = f"{args.color_space.upper()}" + \
                (f", ch:{args.channel}" if args.channel is not None else "") + \
                f", k:{args.k}" + (", inverted" if args.invert else "")
        plot_results(image_rgb, segmented_image, number_mask, title)
        plt.show()
    
    print("Processing complete!")
    print(f"Results saved to:")
    print(f"  Segmented image: {segmented_path}")
    print(f"  Extracted number: {number_path}")
    print(f"  Final visualization: {final_path}")

if __name__ == "__main__":
    main()
