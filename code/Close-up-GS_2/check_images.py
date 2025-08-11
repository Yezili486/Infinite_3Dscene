#!/usr/bin/env python3
"""
Script to check the content of saved images
"""
import cv2
import numpy as np
from pathlib import Path

def check_image(image_path):
    """Check image properties"""
    print(f"\nChecking: {image_path}")
    
    if not Path(image_path).exists():
        print("  File does not exist!")
        return
    
    # File size
    file_size = Path(image_path).stat().st_size
    print(f"  File size: {file_size} bytes")
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print("  Failed to load image!")
            return
        
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min()}, {img.max()}]")
        print(f"  Image mean: {img.mean():.3f}")
        
        # Check if image is all black
        if np.all(img == 0):
            print("  Image is completely black!")
        elif np.all(img == 255):
            print("  Image is completely white!")
        else:
            non_zero_pixels = np.sum(img > 0)
            total_pixels = img.size
            print(f"  Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.2f}%)")
            
        # Color channel stats
        print(f"  Blue channel range: [{img[:,:,0].min()}, {img[:,:,0].max()}]")
        print(f"  Green channel range: [{img[:,:,1].min()}, {img[:,:,1].max()}]")
        print(f"  Red channel range: [{img[:,:,2].min()}, {img[:,:,2].max()}]")
        
    except Exception as e:
        print(f"  Error: {e}")

def main():
    print("=== Checking Saved Images ===")
    
    # Check output directory
    output_dir = Path("real_image_test_output")
    if not output_dir.exists():
        print("Output directory does not exist!")
        return
    
    # Check round directories
    for round_dir in ["round_0", "round_1"]:
        round_path = output_dir / round_dir
        if not round_path.exists():
            print(f"{round_dir} directory does not exist!")
            continue
            
        print(f"\n=== {round_dir.upper()} ===")
        
        # Check all PNG files in the round directory
        for png_file in round_path.glob("*.png"):
            check_image(png_file)

if __name__ == "__main__":
    main()