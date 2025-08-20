#!/usr/bin/env python3
"""
Close-up-GS LLFF Dataset Download Script
Downloads and prepares LLFF dataset for training
"""

import os
import subprocess
import shutil
import sys
import argparse
import zipfile
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def download_from_kaggle():
    """Download LLFF dataset from Kaggle"""
    print("Downloading LLFF dataset from Kaggle...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Try kaggle CLI first
    print("Trying Kaggle CLI download...")
    kaggle_cmd = "kaggle datasets download -d arenagrenade/llff-dataset-full -p data/ --unzip"
    result = run_command(kaggle_cmd)
    
    if result is not None:
        print("Kaggle download completed successfully!")
        return True
    
    # If kaggle CLI fails, try installing it
    print("Kaggle CLI not found or failed. Installing kaggle...")
    install_result = run_command("pip install kaggle")
    if install_result is not None:
        print("Kaggle installed, trying download again...")
        result = run_command(kaggle_cmd)
        if result is not None:
            print("Kaggle download completed successfully!")
            return True
    
    print("Kaggle download failed. Please check your Kaggle API credentials.")
    print("To set up Kaggle API:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json and place it in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    return False

def download_from_google_drive():
    """Download LLFF dataset from Google Drive (fallback)"""
    print("Trying Google Drive download as fallback...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    zip_path = data_dir / "nerf_llff_data.zip"
    
    # Try curl with new folder link
    print("Trying curl download from Google Drive folder...")
    curl_cmd = f'curl -L -o "{zip_path}" "https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"'
    result = run_command(curl_cmd)
    
    if result is None or not zip_path.exists():
        print("Curl failed, trying gdown...")
        # Try gdown with folder ID
        try:
            import gdown
            print("Using gdown to download from folder...")
            gdown.download_folder("https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1", 
                                 output=str(data_dir), quiet=False)
        except ImportError:
            print("gdown not installed. Installing gdown...")
            run_command("pip install gdown")
            import gdown
            gdown.download_folder("https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1", 
                                 output=str(data_dir), quiet=False)
    
    # Check if zip file was downloaded
    if not zip_path.exists():
        print("Failed to download from Google Drive")
        print("\nManual download instructions:")
        print("1. Open https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 in browser")
        print("2. Download nerf_llff_data.zip (~800 MB)")
        print("3. Place the zip file in data/ directory")
        print("4. Run this script again to extract")
        print("5. If download limit reached, try different Google account or VPN")
        return False
    
    # Extract the zip file
    print("Extracting LLFF dataset...")
    llff_dir = data_dir / "llff"
    llff_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(llff_dir)
        print("Extraction completed successfully!")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False
    
    # Clean up zip file
    zip_path.unlink()
    print("Cleaned up zip file")
    
    return True

def download_llff_dataset(force_download=False, keep_repo=False, non_interactive=False):
    """Download and prepare LLFF dataset"""
    print("=== Close-up-GS LLFF Dataset Download ===")
    print("Starting LLFF dataset download...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    llff_dir = data_dir / "llff"
    
    # Check if LLFF directory already exists
    if llff_dir.exists():
        print("Warning: data/llff directory already exists!")
        if force_download:
            print("Force download enabled, removing existing data...")
            shutil.rmtree(llff_dir)
        elif not non_interactive:
            response = input("Do you want to remove existing data and re-download? (y/N): ")
            if response.lower() == 'y':
                print("Removing existing LLFF data...")
                shutil.rmtree(llff_dir)
            else:
                print("Skipping download. Using existing data.")
                return True
        else:
            print("Non-interactive mode: skipping download. Using existing data.")
            return True
    
    # Check if zip file already exists
    zip_path = data_dir / "nerf_llff_data.zip"
    if zip_path.exists():
        print("Found existing nerf_llff_data.zip, extracting...")
        success = extract_zip_file(zip_path, llff_dir)
        if success:
            zip_path.unlink()
            print("Extraction completed successfully!")
        else:
            return False
    else:
        # Try Kaggle download first
        success = download_from_kaggle()
        if not success:
            print("Kaggle download failed, trying Google Drive...")
            success = download_from_google_drive()
        
        if not success:
            print("All download methods failed.")
            print("\nManual download instructions:")
            print("1. Go to https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full")
            print("   OR")
            print("2. Open https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 in browser")
            print("3. Download nerf_llff_data.zip (~800 MB)")
            print("4. Place the zip file in data/ directory")
            print("5. Run this script again to extract")
            print("6. If download limit reached, try different Google account or VPN")
            return False
    
    print("=== LLFF Dataset Download Complete ===")
    print(f"Dataset location: {llff_dir}")
    print()
    
    # List available scenes
    print("Available scenes:")
    if llff_dir.exists():
        for item in llff_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
    
    # Test if poses_bounds.npy exists
    flower_poses = llff_dir / "flower" / "poses_bounds.npy"
    if flower_poses.exists():
        print(f"\nTest: Found poses_bounds.npy in flower/")
    else:
        print(f"\nWarning: poses_bounds.npy not found in flower/")
    
    # Test if images exist
    flower_images = llff_dir / "flower" / "images"
    if flower_images.exists():
        jpg_count = len(list(flower_images.glob('*.jpg')))
        print(f"Test: Found {jpg_count} JPG images in flower/images/")
    
    print()
    print("To use with Close-up-GS, run:")
    print("python train_closeup_gs.py --data_path data/llff/fern --dataset_type llff --target_resolution 256 256")
    
    return True

def extract_zip_file(zip_path, extract_dir):
    """Extract zip file to directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download LLFF dataset for Close-up-GS")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force download even if data exists")
    parser.add_argument("--keep-repo", "-k", action="store_true",
                       help="Keep LLFF repository after download")
    parser.add_argument("--non-interactive", "-n", action="store_true",
                       help="Non-interactive mode (no user prompts)")
    parser.add_argument("--check-only", "-c", action="store_true",
                       help="Only check if dataset exists")
    parser.add_argument("--extract-only", "-e", action="store_true",
                       help="Only extract existing zip file")
    
    args = parser.parse_args()
    
    try:
        if args.check_only:
            llff_dir = Path("data/llff")
            if llff_dir.exists():
                print("LLFF dataset found at data/llff/")
                print("Available scenes:")
                for item in llff_dir.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
                
                # Test poses_bounds.npy
                flower_poses = llff_dir / "flower" / "poses_bounds.npy"
                if flower_poses.exists():
                    print(f"\nTest: Found poses_bounds.npy in flower/")
                else:
                    print(f"\nWarning: poses_bounds.npy not found in flower/")
                
                # Test flower images
                flower_images = llff_dir / "flower" / "images"
                if flower_images.exists():
                    jpg_count = len(list(flower_images.glob('*.jpg')))
                    print(f"Test: Found {jpg_count} JPG images in flower/images/")
                return 0
            else:
                print("LLFF dataset not found. Run download script to download.")
                return 1
        
        if args.extract_only:
            zip_path = Path("data/nerf_llff_data.zip")
            llff_dir = Path("data/llff")
            if zip_path.exists():
                print("Extracting existing zip file...")
                success = extract_zip_file(zip_path, llff_dir)
                if success:
                    zip_path.unlink()
                    print("Extraction completed successfully!")
                    return 0
                else:
                    return 1
            else:
                print("nerf_llff_data.zip not found in data/ directory")
                return 1
        
        success = download_llff_dataset(
            force_download=args.force,
            keep_repo=args.keep_repo,
            non_interactive=args.non_interactive
        )
        
        if success:
            print("\nDownload completed successfully!")
            return 0
        else:
            print("\nDownload failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
