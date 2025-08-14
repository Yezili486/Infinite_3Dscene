#!/usr/bin/env python3
"""
Convenient script to run Close-up-GS training
Optimized for NVIDIA RTX 3070Ti/4090 with batch_size=1
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_training(args):
    """Run Close-up-GS training with optimized settings"""
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training command
    cmd = [
        sys.executable, 'train_closeup_gs.py',
        '--data_path', str(args.data_path),
        '--dataset_type', args.dataset_type,
        '--target_resolution', str(args.width), str(args.height),
        '--config', args.config,
        '--batch_size', '1',  # Optimized for 4090
        '--gpu_id', str(args.gpu_id),
        '--output_dir', str(args.output_dir)
    ]
    
    # Add optional flags
    if args.debug:
        cmd.append('--debug')
    if args.save_intermediate:
        cmd.append('--save_intermediate')
    if args.num_samples:
        cmd.extend(['--num_samples', str(args.num_samples)])
    
    print("=== Close-up-GS Training Command ===")
    print(' '.join(cmd))
    print("=" * 50)
    
    # Set environment variables for NVIDIA GPU optimization
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(args.gpu_id),
        'TORCH_CUDA_ARCH_LIST': '8.6,8.9',  # RTX 3070Ti (8.6) and RTX 4090 (8.9)
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # Deterministic CUDA operations
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',  # Better memory management
    })
    
    # Run training
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("\n🎉 Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Close-up-GS training (RTX 3070Ti/4090 optimized)')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--dataset_type', type=str, choices=['lerf', 'llff', 'nerf', 'synthetic'],
                       default='synthetic', help='Dataset type (default: synthetic)')
    parser.add_argument('--width', type=int, default=512,
                       help='Image width (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                       help='Image height (default: 512)')
    parser.add_argument('--config', type=str, default='config/closeup_gs.yaml',
                       help='Configuration file (default: config/closeup_gs.yaml)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (default: 0)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--num_samples', type=int,
                       help='Number of samples for synthetic dataset')
    
    # Flags
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data_path).exists() and args.dataset_type != 'synthetic':
        print(f"❌ Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Config file does not exist: {args.config}")
        sys.exit(1)
    
    print("=== RTX 3070Ti/4090 Optimized Close-up-GS Training ===")
    print(f"Dataset: {args.dataset_type} from {args.data_path}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Config: {args.config}")
    print(f"GPU: {args.gpu_id}")
    print(f"Output: {args.output_dir}")
    print(f"Debug: {args.debug}")
    print("=" * 50)
    
    # Run training
    success = run_training(args)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
