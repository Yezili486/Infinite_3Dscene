#!/usr/bin/env python3
"""
Close-up-GS: Gaussian Splatting for High-Resolution Close-up View Synthesis
Main training and inference script
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.gaussian_model import GaussianModel
from models.gs_model import GSModel
from models.closeup_refiner import CloseupRefiner
from utils.config import Config
from utils.logger import setup_logger
from train.trainer import Trainer
from train.gs_trainer import GSTrainer
from train.closeup_trainer import CloseupGSTrainer
from data.dataset import CloseUpDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Close-up-GS Training and Inference')
    
    # Basic arguments
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'render'],
                      default='train', help='Mode to run')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Output directory')
    parser.add_argument('--dataset_type', type=str, default='auto', 
                      choices=['lerf', 'llff', 'nerf', 'auto'],
                      help='Dataset type (auto-detect by default)')
    parser.add_argument('--target_resolution', type=int, nargs=2, default=[512, 512],
                      help='Target resolution for preprocessing (width height)')
    parser.add_argument('--model_type', type=str, default='gs', choices=['gs', 'gaussian', 'closeup'],
                      help='Model type: gs (GSModel), gaussian (GaussianModel), or closeup (CloseupGS)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint file')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                      help='GPU ID to use')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting Close-up-GS in {args.mode} mode")
    
    # Load config
    config = Config(args.config)
    
    # Update config with command line arguments
    config.update_from_args(args)
    
    if args.mode == 'train':
        # Training mode
        logger.info("Starting training...")
        
        # Load dataset
        dataset = CloseUpDataset(
            data_path=args.data_path, 
            config=config, 
            split='train',
            dataset_type=args.dataset_type,
            target_resolution=tuple(args.target_resolution)
        )
        
        # Log dataset information
        logger.info(f"Dataset type: {dataset.dataset_type}")
        logger.info(f"Training views: {len(dataset.training_views)}")
        logger.info(f"Close-up test views: {len(dataset.closeup_test_views)}")
        logger.info(f"Object center: {dataset.object_center}")
        
        # Get training views info
        training_info = dataset.get_training_views_info()
        if training_info:
            logger.info(f"Training views distance range: {training_info['min_distance']:.2f} - {training_info['max_distance']:.2f}")
            logger.info(f"Mean training distance: {training_info['mean_distance']:.2f}")
        
        # Initialize model based on type
        if args.model_type == 'closeup':
            trainer = CloseupGSTrainer(dataset, config, device, logger)
            logger.info("Using Complete Close-up-GS System")
        elif args.model_type == 'gs':
            model = GSModel(config).to(device)
            trainer = GSTrainer(model, dataset, config, device, logger)
            logger.info("Using GSModel with GSTrainer for 30K iterations")
        else:
            model = GaussianModel(config).to(device)
            trainer = Trainer(model, dataset, config, device, logger)
            logger.info("Using GaussianModel with standard Trainer")
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        trainer.train()
        
    elif args.mode == 'test':
        # Testing mode
        logger.info("Starting testing...")
        
        if not args.checkpoint:
            raise ValueError("Checkpoint is required for testing mode")
        
        # Load dataset
        dataset = CloseUpDataset(
            data_path=args.data_path, 
            config=config, 
            split='test',
            dataset_type=args.dataset_type,
            target_resolution=tuple(args.target_resolution)
        )
        
        # Initialize model based on type
        if args.model_type == 'gs':
            model = GSModel(config).to(device)
            trainer = GSTrainer(model, dataset, config, device, logger)
        else:
            model = GaussianModel(config).to(device)
            trainer = Trainer(model, dataset, config, device, logger)
        
        # Load checkpoint
        trainer.load_checkpoint(args.checkpoint)
        
        # Start testing
        trainer.test()
        
    elif args.mode == 'render':
        # Rendering mode
        logger.info("Starting rendering...")
        
        if not args.checkpoint:
            raise ValueError("Checkpoint is required for rendering mode")
        
        # Initialize model
        model = GaussianModel(config).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # TODO: Implement rendering logic
        logger.info("Rendering functionality to be implemented")
    
    logger.info("Process completed successfully!")


if __name__ == '__main__':
    main()







