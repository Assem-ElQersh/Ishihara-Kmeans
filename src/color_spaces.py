import numpy as np
import cv2
import matplotlib.pyplot as plt

def convert_to_color_space(image, color_space='rgb'):
    """
    Convert image to specified color space
    
    Args:
        image (numpy.ndarray): Input image in BGR format (as read by cv2)
        color_space (str): Target color space ('rgb', 'hsv', 'lab', 'ycrcb')
        
    Returns:
        numpy.ndarray: Image converted to specified color space
    """
    # Ensure input image is in BGR format (OpenCV default)
    if color_space.lower() == 'rgb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space.lower() == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space.lower() == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space.lower() == 'ycrcb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def visualize_color_channels(image, color_space='rgb'):
    """
    Visualize all channels of an image in the specified color space
    
    Args:
        image (numpy.ndarray): Input image in BGR format (as read by cv2)
        color_space (str): Color space to visualize ('rgb', 'hsv', 'lab', 'ycrcb')
        
    Returns:
        tuple: (figure, converted_image, channel_images, channel_names)
    """
    # Convert to specified color space
    converted = convert_to_color_space(image, color_space)
    
    # Define channel names for each color space
    channel_names = {
        'rgb': ['Red', 'Green', 'Blue'],
        'hsv': ['Hue', 'Saturation', 'Value'],
        'lab': ['Lightness', 'a (Red-Green)', 'b (Blue-Yellow)'],
        'ycrcb': ['Y (Luminance)', 'Cr (Red-Difference)', 'Cb (Blue-Difference)']
    }
    
    # Extract channels
    channels = cv2.split(converted)
    names = channel_names[color_space.lower()]
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Display original image (converted to RGB for display)
    if color_space.lower() != 'rgb':
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = converted
        
    axes[0].imshow(display_image)
    axes[0].set_title(f"Original Image\n({color_space.upper()} color space)")
    axes[0].axis('off')
    
    # Display individual channels
    for i, (channel, name) in enumerate(zip(channels, names)):
        # Normalize for display if needed
        if color_space.lower() in ['hsv', 'lab', 'ycrcb']:
            # Normalize to 0-255 range for better visualization
            normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = channel
            
        axes[i+1].imshow(normalized, cmap='gray')
        axes[i+1].set_title(f"Channel {i}: {name}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig, converted, channels, names

def analyze_channel_histograms(image, color_space='rgb'):
    """
    Analyze histograms of all channels in the specified color space
    
    Args:
        image (numpy.ndarray): Input image in BGR format (as read by cv2)
        color_space (str): Color space to analyze ('rgb', 'hsv', 'lab', 'ycrcb')
        
    Returns:
        tuple: (figure, histograms)
    """
    # Convert to specified color space
    converted = convert_to_color_space(image, color_space)
    
    # Define channel names for each color space
    channel_names = {
        'rgb': ['Red', 'Green', 'Blue'],
        'hsv': ['Hue', 'Saturation', 'Value'],
        'lab': ['Lightness', 'a (Red-Green)', 'b (Blue-Yellow)'],
        'ycrcb': ['Y (Luminance)', 'Cr (Red-Difference)', 'Cb (Blue-Difference)']
    }
    
    # Extract channels
    channels = cv2.split(converted)
    names = channel_names[color_space.lower()]
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Display original image (converted to RGB for display)
    if color_space.lower() != 'rgb':
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = converted
        
    axes[0].imshow(display_image)
    axes[0].set_title(f"Original Image\n({color_space.upper()} color space)")
    axes[0].axis('off')
    
    # Calculate and display histograms
    histograms = []
    for i, (channel, name) in enumerate(zip(channels, names)):
        # Calculate histogram
        if color_space.lower() == 'hsv' and i == 0:  # Special case for HSV hue channel
            hist = cv2.calcHist([channel], [0], None, [180], [0, 180])
        else:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            
        histograms.append(hist)
        
        # Plot histogram
        axes[i+1].plot(hist, color='black')
        axes[i+1].set_xlim([0, 256 if color_space.lower() != 'hsv' or i != 0 else 180])
        axes[i+1].set_title(f"{name} Histogram")
        axes[i+1].grid(True)
    
    plt.tight_layout()
    return fig, histograms

def compare_color_spaces(image):
    """
    Compare all color spaces for an Ishihara test image
    
    Args:
        image (numpy.ndarray): Input image in BGR format (as read by cv2)
        
    Returns:
        figure: Matplotlib figure with color space comparison
    """
    color_spaces = ['rgb', 'hsv', 'lab', 'ycrcb']
    
    # Create figure
    fig, axes = plt.subplots(len(color_spaces), 4, figsize=(16, 4 * len(color_spaces)))
    
    # Original BGR image converted to RGB for display
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i, cs in enumerate(color_spaces):
        # Convert to color space
        converted = convert_to_color_space(image, cs)
        
        # Get channel names
        channel_names = {
            'rgb': ['Red', 'Green', 'Blue'],
            'hsv': ['Hue', 'Saturation', 'Value'],
            'lab': ['Lightness', 'a (Red-Green)', 'b (Blue-Yellow)'],
            'ycrcb': ['Y (Luminance)', 'Cr (Red-Difference)', 'Cb (Blue-Difference)']
        }[cs]
        
        # Extract channels
        channels = cv2.split(converted)
        
        # Display original
        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f"Original in {cs.upper()}")
        axes[i, 0].axis('off')
        
        # Display channels
        for j, (channel, name) in enumerate(zip(channels, channel_names)):
            # Normalize for display if needed
            if cs in ['hsv', 'lab', 'ycrcb']:
                normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            else:
                normalized = channel
                
            axes[i, j+1].imshow(normalized, cmap='gray')
            axes[i, 0].set_title(f"{cs.upper()}")
            axes[i, j+1].set_title(f"{name}")
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    return fig

def recommend_color_space(image):
    """
    Analyze an Ishihara test image and recommend the best color space and channel
    
    Args:
        image (numpy.ndarray): Input image in BGR format (as read by cv2)
        
    Returns:
        dict: Recommendation with color space, channel, and explanation
    """
    # Convert to all color spaces
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Extract channels
    rgb_channels = cv2.split(rgb)
    hsv_channels = cv2.split(hsv)
    lab_channels = cv2.split(lab)
    ycrcb_channels = cv2.split(ycrcb)
    
    # Calculate variance for each channel (higher variance often means better discrimination)
    variances = {
        'rgb_0': np.var(rgb_channels[0]),  # Red
        'rgb_1': np.var(rgb_channels[1]),  # Green
        'rgb_2': np.var(rgb_channels[2]),  # Blue
        'hsv_0': np.var(hsv_channels[0]),  # Hue
        'hsv_1': np.var(hsv_channels[1]),  # Saturation
        'hsv_2': np.var(hsv_channels[2]),  # Value
        'lab_0': np.var(lab_channels[0]),  # Lightness
        'lab_1': np.var(lab_channels[1]),  # a (Red-Green)
        'lab_2': np.var(lab_channels[2]),  # b (Blue-Yellow)
        'ycrcb_0': np.var(ycrcb_channels[0]),  # Y (Luminance)
        'ycrcb_1': np.var(ycrcb_channels[1]),  # Cr (Red-Difference)
        'ycrcb_2': np.var(ycrcb_channels[2]),  # Cb (Blue-Difference)
    }
    
    # Get top variance channels
    sorted_variances = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    
    # Add explanations for each color space and channel
    explanations = {
        'rgb_0': "Red channel shows good discrimination for this Ishihara test",
        'rgb_1': "Green channel shows good discrimination for this Ishihara test",
        'rgb_2': "Blue channel shows good discrimination for this Ishihara test",
        'hsv_0': "Hue channel effectively separates different colors in this test",
        'hsv_1': "Saturation channel shows significant variation in this test",
        'hsv_2': "Value (brightness) channel shows good discrimination in this test",
        'lab_1': "The a channel (red-green axis) is ideal for most Ishihara tests",
        'lab_2': "The b channel (blue-yellow axis) shows good separation for this test",
        'ycrcb_1': "The Cr (red-difference) channel is effective for Ishihara tests",
        'ycrcb_2': "The Cb (blue-difference) channel shows good discrimination"
    }
    
    # Special case: Prioritize Lab a-channel and YCrCb Cr-channel for Ishihara tests
    # based on research findings (they usually work best)
    top_recommendations = []
    
    # Check if Lab a-channel or YCrCb Cr-channel are in top 5
    for cs_ch, _ in sorted_variances[:5]:
        if cs_ch == 'lab_1' or cs_ch == 'ycrcb_1':
            top_recommendations.append(cs_ch)
    
    # If neither is in top 5, add the best of them
    if not top_recommendations:
        if variances['lab_1'] > variances['ycrcb_1']:
            top_recommendations.append('lab_1')
        else:
            top_recommendations.append('ycrcb_1')
    
    # Add the highest variance channel if it's not already included
    if sorted_variances[0][0] not in top_recommendations:
        top_recommendations.append(sorted_variances[0][0])
    
    # Format recommendations
    results = []
    for cs_ch in top_recommendations:
        cs, ch = cs_ch.split('_')
        results.append({
            'color_space': cs,
            'channel': int(ch),
            'variance': variances[cs_ch],
            'explanation': explanations.get(cs_ch, "Shows good statistical properties for this image")
        })
    
    return results
