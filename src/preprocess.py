import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import os
from tqdm import tqdm

def calculate_profiles(image):
    """
    Calculate vertical and horizontal intensity profiles of an image.
    Args:
        image (np.array): 2D array of a single processed image.
    Returns:
        vertical_profile (np.array): Mean intensity for each row.
        horizontal_profile (np.array): Mean intensity for each column.
    """
    vertical_profile = np.mean(image, axis=1)
    horizontal_profile = np.mean(image, axis=0)
    
    # Compute gradients
    vertical_gradient = np.gradient(vertical_profile)
    horizontal_gradient = np.gradient(horizontal_profile)

    # Normalize gradients
    vertical_profile = vertical_gradient / np.max(np.abs(vertical_gradient))
    horizontal_profile = horizontal_gradient / np.max(np.abs(horizontal_gradient))
    
    return vertical_profile, horizontal_profile

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Preprocess a single image with advanced filtering:
    1. Load as grayscale
    2. Apply statistical filtering
    3. Apply Gaussian blur
    4. Perform morphological operations
    5. Apply edge detection
    6. Resize to target size
    7. Normalize to [0,1] range
    8. Calculate intensity profiles
    """
    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Statistical filtering
    std = np.std(img)
    mean = np.mean(img)
    img_clipped = np.clip(img, mean + (2.5*std), mean + (5*std))
    
    # Apply Gaussian blur
    gaussian = cv2.GaussianBlur(img_clipped, (3, 3), 5)
    
    # Morphological operations
    kernel3 = np.ones((3, 3), dtype=np.float32)
    kernel5 = np.ones((5, 5), dtype=np.float32)
    morphed = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel=kernel3)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel=kernel5)
    
    # Edge detection
    sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 10)
    sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 10)
    blended = cv2.addWeighted(src1=sobelx, alpha=0.9, 
                            src2=sobely, beta=0.3, gamma=0.25)
    
    # Resize
    img_resized = cv2.resize(blended, target_size)
    
    # Normalize to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Calculate intensity profiles
    vertical_profile, horizontal_profile = calculate_profiles(img_normalized)
    
    return img_normalized, vertical_profile, horizontal_profile

def visualize_preprocessing_steps(img_path, target_size=(128, 128)):
    """
    Visualize each step of the preprocessing pipeline for a single image.
    Args:
        img_path (str or Path): Path to the input image
        target_size (tuple): Target size for resizing (width, height)
    """
    # Load original image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Statistical filtering
    std = np.std(img)
    mean = np.mean(img)
    img_clipped = np.clip(img, mean + (2.5*std), mean + (5*std))
    
    # Apply Gaussian blur
    gaussian = cv2.GaussianBlur(img_clipped, (3, 3), 5)
    
    # Morphological operations
    kernel3 = np.ones((3, 3), dtype=np.float32)
    kernel5 = np.ones((5, 5), dtype=np.float32)
    morphed = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel=kernel3)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel=kernel5)
    
    # Edge detection
    sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 10)
    sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 10)
    blended = cv2.addWeighted(src1=sobelx, alpha=0.9, 
                            src2=sobely, beta=0.3, gamma=0.25)
    
    # Resize
    img_resized = cv2.resize(blended, target_size)
    
    # Normalize to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Plot all steps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    steps = [('Original', img),
             ('Statistical Filtering', img_clipped),
             ('Gaussian Blur', gaussian),
             ('Morphological Ops', morphed),
             ('Edge Detection', blended),
             ('Resized', img_resized),
             ('Normalized', img_normalized)]
    
    for (title, img_step), ax in zip(steps, axes.ravel()):
        ax.imshow(img_step, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def preprocess_directory(input_dir, output_dir, target_size=(128, 128)):
    """
    Preprocess all images in a directory and save to output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create profiles directory
    profiles_dir = output_path.parent / 'profiles'
    profiles_dir.mkdir(exist_ok=True)
    
    # Process each image
    for img_path in tqdm(list(input_path.glob('*.png')), desc=f"Processing {input_path.name}"):
        try:
            # Preprocess image
            processed_img, vertical_profile, horizontal_profile = preprocess_image(img_path, target_size)
            
            # Save processed image
            output_img_path = output_path / img_path.name
            cv2.imwrite(str(output_img_path), (processed_img * 255).astype(np.uint8))
            
            # Save profiles
            profile_path = profiles_dir / f"{img_path.stem}_profiles.npz"
            np.savez(str(profile_path), 
                    vertical=vertical_profile,
                    horizontal=horizontal_profile)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

def preprocess_all_data():
    """
    Preprocess all data in the raw directory
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent  # Adjust the number of .parent calls as needed
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    
    # Process each signal type directory
    for signal_type_dir in raw_dir.iterdir():
        if signal_type_dir.is_dir():
            print(f"Processing {signal_type_dir.name}...")
            output_dir = processed_dir / signal_type_dir.name
            preprocess_directory(signal_type_dir, output_dir)

if __name__ == '__main__':
    preprocess_all_data() 