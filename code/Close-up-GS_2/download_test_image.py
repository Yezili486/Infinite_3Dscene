#!/usr/bin/env python3
"""
Download a test image for Close-up-GS testing
"""

import requests
import os
from pathlib import Path

def download_test_image():
    """Download a test image from Unsplash"""
    
    # Create test_images directory if it doesn't exist
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # URLs for different test images
    image_urls = [
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",  # Mountain landscape
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",  # Forest
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",  # Another landscape
    ]
    
    for i, url in enumerate(image_urls):
        try:
            print(f"Downloading image {i+1}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the image
            filename = test_dir / f"test_image_{i+1}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded: {filename}")
            
        except Exception as e:
            print(f"Failed to download image {i+1}: {e}")
    
    print(f"\nTest images saved in: {test_dir.absolute()}")

if __name__ == "__main__":
    download_test_image() 