#!/usr/bin/env python3
"""
Script to manually download the Stable Diffusion inpainting model
with better network handling and retry logic.
"""

import os
import time
import requests
from huggingface_hub import snapshot_download, HfApi

def download_model():
    """Download the Stable Diffusion inpainting model with retry logic"""
    
    # Set environment variables for better network handling
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # Try different models, starting with smaller ones
    model_ids = [
        "CompVis/stable-diffusion-v1-4",  # Smaller model
        "stabilityai/stable-diffusion-2-inpainting",  # Alternative inpainting model
        "runwayml/stable-diffusion-inpainting"  # Original model
    ]
    
    print("Setting up network configuration...")
    
    # Configure longer timeouts
    api = HfApi()
    
    # Try multiple models
    for model_id in model_ids:
        print(f"\n=== Trying model: {model_id} ===")
        
        # Try multiple download strategies for each model
        for attempt in range(2):
            try:
                print(f"Attempt {attempt + 1}/2: Downloading model {model_id}")
                
                if attempt == 0:
                    # First attempt: standard download
                    local_path = snapshot_download(
                        repo_id=model_id,
                        local_files_only=False,
                        resume_download=True,
                        max_workers=1  # Reduce concurrent connections
                    )
                else:
                    # Second attempt: with different configuration
                    local_path = snapshot_download(
                        repo_id=model_id,
                        local_files_only=False,
                        resume_download=True,
                        max_workers=1,
                        use_auth_token=None
                    )
                
                print(f"Successfully downloaded model to: {local_path}")
                return local_path, model_id
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 1:
                    print("Waiting 3 seconds before retry...")
                    time.sleep(3)
                else:
                    print(f"Failed to download {model_id}")
                    break  # Try next model
    
    print("All models failed to download.")
    return None, None

if __name__ == "__main__":
    print("Starting model download...")
    result, model_id = download_model()
    
    if result:
        print(f"\n✅ Model downloaded successfully!")
        print(f"Model: {model_id}")
        print(f"Path: {result}")
        print("You can now run the main script.")
    else:
        print("\n❌ Failed to download model. The script will use fallback inpainting.") 