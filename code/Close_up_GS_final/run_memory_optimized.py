#!/usr/bin/env python3
"""
Memory-optimized training script for RTX 3070Ti (8GB)
Extreme memory optimization for Close-up-GS training
"""

import os
import sys
import subprocess
import time

def setup_memory_environment():
    """Setup environment variables for optimal memory usage"""
    # CUDA memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable blocking for better performance
    os.environ['CUDA_CACHE_DISABLE'] = '0'    # Keep cache enabled
    
    # PyTorch optimizations
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3070Ti architecture
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    print("Environment configured for RTX 3070Ti memory optimization")

def run_training():
    """Run training with minimal memory usage"""
    setup_memory_environment()
    
    # Ultra-conservative parameters for 8GB GPU
    cmd = [
        'python', 'train_closeup_gs.py',
        '--data_path', './test_data',
        '--dataset_type', 'synthetic', 
        '--target_resolution', '256', '256',  # Keep 256x256 but reduce samples
        '--debug',
        '--num_samples', '3',  # Reduced from 5 to 3
        '--config', 'config/debug_gs.yaml',
        '--output_dir', './outputs'
    ]
    
    print("Starting memory-optimized training...")
    print("GPU: RTX 3070Ti (8GB)")
    print("Target resolution: 256x256")
    print("Samples: 3 (reduced)")
    print("Baseline iterations: 100")
    print("-" * 50)
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\nTraining completed successfully!")
        else:
            print(f"\nTraining failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    run_training()
