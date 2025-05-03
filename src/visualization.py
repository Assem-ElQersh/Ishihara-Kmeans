import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

def plot_results(original_image, segmented_image, number_mask, title=None):
    """
    Plot the original image, segmented image, and extracted number mask
    
    Args:
        original_image (numpy.ndarray): Original RGB image
        segmented_image (numpy.ndarray): Segmented image
        number_mask (numpy.ndarray): Extracted number mask
        title (str, optional): Title for the plot
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    if len(segmented_image.shape) == 2:
        plt.imshow(segmented_image, cmap="gray")
    else:
        plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(number_mask, cmap="gray")
    plt.title("Extracted Number")
    plt.axis("off")
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()

def plot_color_space_comparison(original_image, results):
    """
    Plot comparison of different color spaces and parameters
    
    Args:
        original_image (numpy.ndarray): Original RGB image
        results (list): List of tuples (parameters, segmented_image, number_mask)
    """
    n_results = len(results)
    fig = plt.figure(figsize=(15, 4 * n_results))
    gs = gridspec.GridSpec(n_results, 3)
    
    for i, (params, segmented_image, number_mask) in enumerate(results):
        # Plot original image
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # Plot segmented image
        ax2 = plt.subplot(gs[i, 1])
        if len(segmented_image.shape) == 2:
            ax2.imshow(segmented_image, cmap="gray")
        else:
            ax2.imshow(segmented_image)
        
        title = params['title']
        details = f"{params['color_space'].upper()}"
        if params['channel'] is not None:
            details += f", ch:{params['channel']}"
        details += f", k:{params['k']}"
        if params['invert']:
            details += ", inverted"
        
        ax2.set_title(f"{title}\n({details})")
        ax2.axis("off")
        
        # Plot extracted number
        ax3 = plt.subplot(gs[i, 2])
        ax3.imshow(number_mask, cmap="gray")
        ax3.set_title("Extracted Number")
        ax3.axis("off")
    
    plt.tight_layout()
    return fig

def save_comparison_figure(original_image, results, output_path="color_space_comparison.png"):
    """
    Save a comparison figure of different color spaces and parameters
    
    Args:
        original_image (numpy.ndarray): Original RGB image
        results (list): List of tuples (parameters, segmented_image, number_mask)
        output_path (str): Path to save the figure
    """
    fig = plot_color_space_comparison(original_image, results)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure saved to {output_path}")

def create_visualization_grid(images, titles, cols=3, figure_size=(15, 10)):
    """
    Create a grid of images with titles
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image
        cols (int): Number of columns in the grid
        figure_size (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    assert len(images) == len(titles), "Number of images must match number of titles"
    
    # Calculate number of rows needed
    rows = (len(images) + cols - 1) // cols
    
    fig = plt.figure(figsize=figure_size)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Display image (handle grayscale vs. color)
        if len(image.shape) == 2 or image.shape[2] == 1:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
            
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def enhance_number_visibility(number_mask):
    """
    Enhance the visibility of the extracted number with contour detection
    
    Args:
        number_mask (numpy.ndarray): Binary mask of extracted number
        
    Returns:
        numpy.ndarray: Enhanced number visualization
    """
    # Ensure the mask is binary
    binary_mask = (number_mask > 127).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a colored visualization
    vis_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    
    # Draw filled contours in green
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), -1)
    
    # Draw contours in white
    cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)
    
    return vis_image

def create_final_visualization(original_image, number_mask, output_path="final_result.png"):
    """
    Create a side-by-side visualization of original image and extracted number
    
    Args:
        original_image (numpy.ndarray): Original RGB image
        number_mask (numpy.ndarray): Extracted number mask
        output_path (str): Path to save the visualization
    """
    # Enhance the number visibility
    enhanced_number = enhance_number_visibility(number_mask)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.set_title("Original Ishihara Test")
    ax1.axis("off")
    
    # Display enhanced number
    ax2.imshow(enhanced_number)
    ax2.set_title("Extracted Number")
    ax2.axis("off")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Final visualization saved to {output_path}")
    
    return enhanced_number
