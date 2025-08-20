#!/usr/bin/env python3
"""
测试LLFF数据集加载功能
验证数据格式和兼容性
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import CloseUpDataset
from utils.config import Config

def test_llff_loading():
    """测试LLFF数据集加载"""
    print("=== LLFF Dataset Loading Test ===")
    
    # Test with downloaded LLFF data  
    data_path = "./data/nerf_llff_data/flower"
    
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found. Please run download_llff.py first.")
        return False
    
    try:
        # Create dummy config
        config = Config()
        
        print(f"\nTesting LLFF dataset loading from: {data_path}")
        
        # Test train split
        print("\n--- Testing Train Split ---")
        dataset = CloseUpDataset(
            data_path=data_path,
            config=config,
            split='train',
            dataset_type='llff',
            target_resolution=(512, 512)
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Pose shape: {sample['pose'].shape}")
            print(f"Pose 3x5 shape: {sample['pose_3x5'].shape}")
            print(f"Focal: {sample['focal']}")
            print(f"Object center: {sample['object_center']}")
            print(f"Image size: [{sample['image_width']}, {sample['image_height']}]")
            
            # Save test image
            image_np = sample['image'].permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            cv2.imwrite('test_llff.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print(f"Saved test image: test_llff.jpg")
            
            # Test tuple format extraction (for compatibility)
            img = sample['image']
            pose = sample['pose_3x5'] 
            focal = sample['focal']
            print(f"\nLLFF tuple format test:")
            print(f"img.shape: {img.shape}, pose.shape: {pose.shape}, focal: {focal}")
            
            # Test multiple samples
            print(f"\n--- Testing Multiple Samples ---")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: image {sample['image'].shape}, focal {sample['focal']:.2f}")
        
        # Test test split
        print("\n--- Testing Test Split ---")
        test_dataset = CloseUpDataset(
            data_path=data_path,
            config=config,
            split='test',
            dataset_type='llff',
            target_resolution=(512, 512)
        )
        
        print(f"Test dataset length: {len(test_dataset)}")
        
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"Test sample image shape: {sample['image'].shape}")
            print(f"Test sample focal: {sample['focal']}")
        
        print("\n✓ LLFF loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ LLFF loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_llff_data():
    """测试真实LLFF数据（如果可用）"""
    print("\n=== Real LLFF Data Test ===")
    
    # Test with real LLFF data if available
    real_llff_paths = [
        "./data/llff/flower",
        "./data/llff/fern", 
        "./nerf_llff_data/flower",
        "./nerf_llff_data/fern"
    ]
    
    for data_path in real_llff_paths:
        if Path(data_path).exists() and (Path(data_path) / "poses_bounds.npy").exists():
            print(f"\nFound real LLFF data at: {data_path}")
            
            try:
                config = Config()
                dataset = CloseUpDataset(
                    data_path=data_path,
                    config=config,
                    split='train',
                    dataset_type='llff',
                    target_resolution=(256, 256)  # Use smaller resolution for real data
                )
                
                print(f"Real LLFF dataset length: {len(dataset)}")
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"Real image shape: {sample['image'].shape}")
                    print(f"Real focal: {sample['focal']:.2f}")
                    print(f"Real object center: {sample['object_center']}")
                    
                    # Save real test image
                    image_np = sample['image'].permute(1, 2, 0).numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    scene_name = Path(data_path).name
                    cv2.imwrite(f'test_real_llff_{scene_name}.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                    print(f"Saved real test image: test_real_llff_{scene_name}.jpg")
                
                print(f"✓ Real LLFF data test passed for {data_path}")
                return True
                
            except Exception as e:
                print(f"✗ Real LLFF data test failed for {data_path}: {e}")
                continue
    
    print("No real LLFF data found or all tests failed")
    return False

def main():
    """主函数"""
    print("LLFF Dataset Loading Test")
    print("=" * 50)
    
    # Test with generated LLFF data
    success1 = test_llff_loading()
    
    # Test with real LLFF data if available
    success2 = test_with_real_llff_data()
    
    if success1:
        print("\n🎉 LLFF dataset loading functionality is working!")
        print("\nUsage example:")
        print("dataset = CloseUpDataset(data_path='./llff_data', dataset_type='llff', split='train')")
        print("img, pose, focal = dataset[0]['image'], dataset[0]['pose_3x5'], dataset[0]['focal']")
        print("\nTo train with LLFF data:")
        print("python train_closeup_gs.py --data_path ./llff_data --dataset_type llff --target_resolution 256 256")
        
        if success2:
            print("\n✓ Both generated and real LLFF data tests passed!")
        else:
            print("\n⚠ Generated LLFF test passed, but no real LLFF data available")
    else:
        print("\n❌ LLFF dataset loading test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
