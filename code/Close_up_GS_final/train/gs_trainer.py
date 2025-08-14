"""
Gaussian Splatting Trainer for Initial Model Training
Optimizes from training views Vs for 30000 iterations (Paper Section 5.2)
"""

import torch
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import time
import random

from models.gs_model import GSModel
from data.dataset import CloseUpDataset
from utils.logger import MetricsLogger, ProgressLogger


class GSTrainer:
    """
    Trainer for initial Gaussian Splatting model
    Implements optimization from training views Vs for 30000 iterations
    """
    
    def __init__(self, model: GSModel, dataset: CloseUpDataset, config, device, logger):
        """
        Initialize trainer
        
        Args:
            model: GSModel instance
            dataset: CloseUpDataset with training views
            config: Configuration object
            device: Training device
            logger: Logger instance
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger
        
        # Training parameters from paper Section 5.2
        self.max_iterations = 30000  # 30K iterations as specified in paper
        self.learning_rates = {
            'position': 0.00016,
            'position_final': 0.0000016,
            'feature': 0.0025,
            'opacity': 0.05,
            'scaling': 0.005,
            'rotation': 0.001
        }
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger(
            config.get('output_dir', './outputs'),
            use_tensorboard=True,
            use_wandb=config.get('use_wandb', False)
        )
        
        # Training state
        self.current_iteration = 0
        self.best_loss = float('inf')
        
        # Training views (Vs) - far distance views for initial training
        self.training_views = dataset.training_views
        self.logger.info(f"Training with {len(self.training_views)} far distance views (Vs)")
        
        # Densification schedule
        self.densify_from_iter = config.get('densify_from_iter', 500)
        self.densify_until_iter = config.get('densify_until_iter', 15000)
        self.densification_interval = config.get('densification_interval', 100)
        self.opacity_reset_interval = config.get('opacity_reset_interval', 3000)
        
        # Progress tracking
        self.progress_logger = ProgressLogger(self.max_iterations, log_interval=500)
    
    def setup_optimizers(self):
        """Setup optimizers for different parameter groups"""
        if self.model.get_centers.shape[0] == 0:
            self.logger.warning("No Gaussians initialized, skipping optimizer setup")
            return
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': [self.model._centers], 'lr': self.learning_rates['position'], 'name': "xyz"},
            {'params': [self.model._sh_coeffs], 'lr': self.learning_rates['feature'], 'name': "f_sh"},
            {'params': [self.model._opacities], 'lr': self.learning_rates['opacity'], 'name': "opacity"},
            {'params': [self.model._scales], 'lr': self.learning_rates['scaling'], 'name': "scaling"},
            {'params': [self.model._rotations], 'lr': self.learning_rates['rotation'], 'name': "rotation"}
        ]
        
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # Learning rate scheduler for position (exponential decay)
        self.position_lr_scheduler = self.get_expon_lr_func(
            lr_init=self.learning_rates['position'],
            lr_final=self.learning_rates['position_final'],
            max_steps=self.max_iterations
        )
    
    def get_expon_lr_func(self, lr_init: float, lr_final: float, max_steps: int):
        """Get exponential learning rate scheduler function"""
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return 0.0
            t = min(step / max_steps, 1.0)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return log_lerp
        return helper
    
    def update_learning_rate(self, iteration: int):
        """Update learning rates"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.position_lr_scheduler(iteration)
                param_group['lr'] = lr
    
    def train(self):
        """
        Main training loop - optimize from training views Vs for 30000 iterations
        """
        self.logger.info(f"Starting training for {self.max_iterations} iterations")
        self.logger.info(f"Using {len(self.training_views)} training views (far distance)")
        
        # Initialize point cloud if not done
        if self.model.get_centers.shape[0] == 0:
            self.initialize_from_training_views()
        
        # Setup optimizers after initialization
        if not hasattr(self, 'optimizer'):
            self.setup_optimizers()
        
        self.model.train()
        start_time = time.time()
        
        for iteration in range(self.current_iteration, self.max_iterations):
            self.current_iteration = iteration
            
            # Update learning rates
            self.update_learning_rate(iteration)
            
            # Training step
            loss_dict = self.training_step()
            
            # Log metrics
            if iteration % 100 == 0:
                self.log_metrics(loss_dict, iteration)
                
            # Progress update
            if iteration % 500 == 0:
                elapsed = time.time() - start_time
                self.progress_logger.update(iteration, loss_dict)
                self.logger.info(
                    f"Iter {iteration}/{self.max_iterations} | "
                    f"Loss: {loss_dict['total_loss']:.6f} | "
                    f"Gaussians: {loss_dict['num_gaussians']} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Densification (only when distance < 1/3)
            if (iteration >= self.densify_from_iter and 
                iteration <= self.densify_until_iter and
                iteration % self.densification_interval == 0):
                
                # Get current camera distance to scene center
                camera_distance = self.get_current_camera_distance()
                self.model.densify(camera_distance)
            
            # Reset opacity periodically
            if (iteration % self.opacity_reset_interval == 0 and 
                iteration > 0):
                self.model.reset_opacity()
                self.logger.info(f"Reset opacity at iteration {iteration}")
            
            # Pruning
            if iteration % 1000 == 0 and iteration > 0:
                self.model.prune_gaussians()
            
            # Save checkpoint
            if iteration % 5000 == 0 and iteration > 0:
                self.save_checkpoint(iteration, loss_dict['total_loss'])
            
            # Update best loss
            if loss_dict['total_loss'] < self.best_loss:
                self.best_loss = loss_dict['total_loss']
                if iteration % 1000 == 0:  # Save best model periodically
                    self.save_checkpoint(iteration, loss_dict['total_loss'], is_best=True)
        
        # Final save
        self.save_checkpoint(self.max_iterations, loss_dict['total_loss'], is_best=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Final loss: {self.best_loss:.6f}")
        self.logger.info(f"Final Gaussians: {self.model.get_centers.shape[0]}")
        
        self.metrics_logger.close()
    
    def initialize_from_training_views(self):
        """Initialize point cloud from training views"""
        self.logger.info("Initializing point cloud from training views...")
        
        # Simple initialization: sample points from camera frustums
        all_points = []
        all_colors = []
        
        # Sample a few training views for initialization
        sample_views = random.sample(self.training_views, min(5, len(self.training_views)))
        
        for view_idx in sample_views:
            sample = self.dataset[view_idx]
            camera = sample['camera']
            
            # Generate random points in front of camera
            num_points = 1000
            depth_range = [1.0, 10.0]
            
            # Random points in camera space
            points_cam = torch.randn(num_points, 3) * 2.0
            points_cam[:, 2] = torch.rand(num_points) * (depth_range[1] - depth_range[0]) + depth_range[0]
            
            # Transform to world space
            points_homo = torch.cat([points_cam, torch.ones(num_points, 1)], dim=1)
            points_world = (camera.camera_to_world @ points_homo.T).T[:, :3]
            
            # Random colors
            colors = torch.rand(num_points, 3)
            
            all_points.append(points_world)
            all_colors.append(colors)
        
        # Combine all points
        if all_points:
            points = torch.cat(all_points, dim=0)
            colors = torch.cat(all_colors, dim=0)
            
            # Move to device
            points = points.to(self.device)
            colors = colors.to(self.device)
            
            # Initialize model
            self.model.create_from_point_cloud(points, colors)
            self.logger.info(f"Initialized {points.shape[0]} Gaussians from training views")
        else:
            self.logger.error("Failed to initialize point cloud")
    
    def training_step(self) -> Dict[str, float]:
        """Single training step"""
        # Sample random training view
        view_idx = random.choice(self.training_views)
        sample = self.dataset[view_idx]
        
        camera = sample['camera']
        target_image = sample['image'].to(self.device)
        
        # Forward pass
        rendered_output = self.model(camera)
        
        # Compute loss
        loss_dict = self.model.compute_loss(rendered_output, target_image)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Update training statistics for densification
        if self.model._centers.grad is not None:
            self.model.update_training_stats(self.model._centers.grad)
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert to float for logging
        result = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.item()
            else:
                result[key] = value
        
        return result
    
    def get_current_camera_distance(self) -> float:
        """Get current camera distance to scene center"""
        # Sample current training view
        view_idx = random.choice(self.training_views)
        sample = self.dataset[view_idx]
        camera = sample['camera']
        
        # Distance to object center
        distance = torch.norm(camera.camera_center - sample['object_center'])
        return distance.item()
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log training metrics"""
        for key, value in metrics.items():
            self.metrics_logger.log_scalar(f"train/{key}", value, iteration)
        
        # Log learning rates
        for param_group in self.optimizer.param_groups:
            lr_name = f"lr/{param_group['name']}"
            self.metrics_logger.log_scalar(lr_name, param_group['lr'], iteration)
    
    def save_checkpoint(self, iteration: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.to_dict(),
            'training_views': self.training_views,
            'best_loss': self.best_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.get('output_dir', './outputs')) / f'gs_checkpoint_iter_{iteration}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.get('output_dir', './outputs')) / 'gs_best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_iteration = checkpoint['iteration']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from iteration {self.current_iteration}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test views"""
        self.logger.info("Evaluating model...")
        
        self.model.eval()
        total_loss = 0.0
        total_l1 = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        # Evaluate on close-up test views
        closeup_samples = self.dataset.get_closeup_test_samples()
        
        with torch.no_grad():
            for sample in tqdm(closeup_samples, desc="Evaluating"):
                camera = sample['camera']
                target_image = sample['image'].to(self.device)
                
                # Forward pass
                rendered_output = self.model(camera)
                loss_dict = self.model.compute_loss(rendered_output, target_image)
                
                total_loss += loss_dict['total_loss'].item()
                total_l1 += loss_dict['l1_loss'].item()
                total_ssim += loss_dict['ssim_loss'].item()
                num_samples += 1
        
        # Compute averages
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_l1 = total_l1 / num_samples if num_samples > 0 else 0.0
        avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
        
        results = {
            'eval_loss': avg_loss,
            'eval_l1': avg_l1,
            'eval_ssim': avg_ssim,
            'num_test_views': num_samples
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Average Loss: {avg_loss:.6f}")
        self.logger.info(f"  Average L1: {avg_l1:.6f}")
        self.logger.info(f"  Average SSIM: {avg_ssim:.6f}")
        self.logger.info(f"  Test Views: {num_samples}")
        
        self.model.train()
        return results
    
    def render_test_images(self, num_images: int = 10):
        """Render test images for visualization"""
        self.logger.info(f"Rendering {num_images} test images...")
        
        self.model.eval()
        output_dir = Path(self.config.get('output_dir', './outputs')) / 'rendered_images'
        output_dir.mkdir(exist_ok=True)
        
        # Get close-up test samples
        closeup_samples = self.dataset.get_closeup_test_samples()
        samples_to_render = closeup_samples[:num_images]
        
        with torch.no_grad():
            for i, sample in enumerate(samples_to_render):
                camera = sample['camera']
                target_image = sample['image']
                
                # Render image
                rendered_output = self.model(camera)
                rendered_image = rendered_output['image']
                
                # Save images (simplified - in practice you'd use proper image saving)
                self.logger.info(f"Rendered test image {i+1}/{len(samples_to_render)}")
        
        self.model.train()


if __name__ == '__main__':
    # Test GSTrainer
    from utils.config import Config
    from models.gs_model import GSModel
    from data.dataset import SyntheticDataset
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic dataset for testing
    dataset = SyntheticDataset(num_samples=50)
    model = GSModel(config).to(device)
    
    # Simple logger
    import logging
    logger = logging.getLogger('test')
    
    trainer = GSTrainer(model, dataset, config, device, logger)
    print("GSTrainer implementation completed!")
