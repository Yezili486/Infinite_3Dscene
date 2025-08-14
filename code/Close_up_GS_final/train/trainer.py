"""
Training logic for Close-up-GS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json

from utils.logger import MetricsLogger, ProgressLogger


class Trainer:
    """Trainer class for Close-up-GS model"""
    
    def __init__(self, model, dataset, config, device, logger):
        """
        Initialize trainer
        
        Args:
            model: Gaussian model to train
            dataset: Training dataset
            config: Configuration object
            device: Training device
            logger: Logger instance
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger(
            config.get('output_dir', './outputs'),
            use_tensorboard=True,
            use_wandb=config.get('use_wandb', False)
        )
        
        # Training state
        self.current_epoch = 0
        self.current_iter = 0
        self.best_loss = float('inf')
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def setup_optimizer(self):
        """Setup optimizers for different parameter groups"""
        # Separate parameter groups with different learning rates
        param_groups = [
            {'params': [self.model._xyz], 'lr': self.config.position_lr_init, 'name': "xyz"},
            {'params': [self.model._features_dc], 'lr': self.config.feature_lr, 'name': "f_dc"},
            {'params': [self.model._features_rest], 'lr': self.config.feature_lr / 20.0, 'name': "f_rest"},
            {'params': [self.model._opacity], 'lr': self.config.opacity_lr, 'name': "opacity"},
            {'params': [self.model._scaling], 'lr': self.config.scaling_lr, 'name': "scaling"},
            {'params': [self.model._rotation], 'lr': self.config.rotation_lr, 'name': "rotation"}
        ]
        
        # Add detail enhancer parameters if available
        if hasattr(self.model, 'detail_enhancer'):
            param_groups.append({
                'params': list(self.model.detail_enhancer.parameters()),
                'lr': self.config.learning_rate,
                'name': "detail_enhancer"
            })
        
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # Learning rate scheduler for xyz (position) parameters
        self.xyz_scheduler = self.get_expon_lr_func(
            lr_init=self.config.position_lr_init,
            lr_final=self.config.position_lr_final,
            lr_delay_mult=self.config.position_lr_delay_mult,
            max_steps=self.config.position_lr_max_steps
        )
    
    def get_expon_lr_func(self, lr_init, lr_final, lr_delay_mult, max_steps):
        """Get exponential learning rate scheduler function"""
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return 0.0
            if lr_delay_mult < 1:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / max_steps, 0, 1))
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp
        return helper
    
    def update_learning_rate(self, iteration):
        """Update learning rates"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(iteration)
                param_group['lr'] = lr
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        # Setup data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup progress logger
        total_iterations = len(dataloader) * self.config.max_epochs
        progress_logger = ProgressLogger(total_iterations)
        
        self.model.train()
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                self.current_iter += 1
                
                # Update learning rates
                self.update_learning_rate(self.current_iter)
                
                # Forward pass
                loss, metrics = self.train_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log metrics
                if self.current_iter % 100 == 0:
                    self.log_metrics(metrics, self.current_iter)
                    progress_logger.update(self.current_iter, metrics)
                
                # Densification (gaussian splitting/cloning)
                if (self.current_iter >= self.config.densify_from_iter and 
                    self.current_iter <= self.config.densify_until_iter and
                    self.current_iter % self.config.densification_interval == 0):
                    self.densify_and_prune()
                
                # Reset opacity
                if (self.current_iter % self.config.opacity_reset_interval == 0 or
                    (self.config.white_background and self.current_iter == self.config.densify_from_iter)):
                    self.reset_opacity()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, avg_epoch_loss)
            
            # Update best loss
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint(epoch + 1, avg_epoch_loss, is_best=True)
        
        self.logger.info("Training completed!")
        self.metrics_logger.close()
    
    def train_step(self, batch) -> Tuple[torch.Tensor, Dict]:
        """Single training step"""
        # Get batch data
        viewpoint_camera = batch['camera']
        gt_image = batch['image'].to(self.device)
        
        # Forward pass through model
        gaussian_outputs = self.model(viewpoint_camera)
        
        # Render image using gaussian splatting
        rendered_image = self.render_gaussians(gaussian_outputs, viewpoint_camera)
        
        # Compute losses
        l1_loss = self.l1_loss(rendered_image, gt_image)
        ssim_loss = 1.0 - self.ssim(rendered_image, gt_image)
        
        # Total loss
        total_loss = (1.0 - self.config.get('lambda_dssim', 0.2)) * l1_loss + self.config.get('lambda_dssim', 0.2) * ssim_loss
        
        # Metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/l1': l1_loss.item(),
            'loss/ssim': ssim_loss.item(),
            'stats/num_gaussians': gaussian_outputs['xyz'].shape[0],
            'lr/xyz': self.optimizer.param_groups[0]['lr']
        }
        
        return total_loss, metrics
    
    def render_gaussians(self, gaussian_outputs, viewpoint_camera):
        """Render image from gaussian outputs (placeholder)"""
        # This would typically call the CUDA rasterization kernel
        # For now, return a dummy tensor with correct shape
        batch_size = 1
        height, width = viewpoint_camera.image_height, viewpoint_camera.image_width
        return torch.randn(batch_size, 3, height, width, device=self.device)
    
    def ssim(self, img1, img2):
        """Compute SSIM between two images (placeholder)"""
        # Simplified SSIM computation
        return torch.tensor(0.8, device=self.device)
    
    def densify_and_prune(self):
        """Densify and prune gaussians"""
        # Store gradients for xyz parameters
        grads = self.model._xyz.grad
        
        if grads is not None:
            # Find gaussians that need densification (high gradient)
            grad_threshold = self.config.densify_grad_threshold
            candidates = torch.norm(grads, dim=-1) >= grad_threshold
            
            if candidates.sum() > 0:
                self.logger.info(f"Densifying {candidates.sum()} gaussians")
                # Simplified densification logic
                # In practice, this would involve splitting/cloning gaussians
                
        # Prune gaussians with low opacity
        opacity_threshold = self.config.min_opacity
        prune_mask = self.model.get_opacity.squeeze() < opacity_threshold
        
        if prune_mask.sum() > 0:
            self.logger.info(f"Pruning {prune_mask.sum()} gaussians")
            # In practice, this would remove low-opacity gaussians
    
    def reset_opacity(self):
        """Reset opacity of gaussians"""
        opacities_new = self.model.opacity_activation(
            torch.ones_like(self.model._opacity) * 0.01
        )
        # Reset opacity values (simplified)
        with torch.no_grad():
            self.model._opacity.data = opacities_new.data
    
    def log_metrics(self, metrics: Dict, iteration: int):
        """Log training metrics"""
        for key, value in metrics.items():
            self.metrics_logger.log_scalar(key, value, iteration)
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.to_dict(),
            'current_iter': self.current_iter
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.get('output_dir', './outputs')) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.get('output_dir', './outputs')) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_iter = checkpoint.get('current_iter', 0)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, iteration {self.current_iter}")
    
    def test(self):
        """Evaluation on test set"""
        self.logger.info("Starting evaluation...")
        
        # Setup data loader for test set
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
                # Forward pass
                viewpoint_camera = batch['camera']
                gt_image = batch['image'].to(self.device)
                
                gaussian_outputs = self.model(viewpoint_camera)
                rendered_image = self.render_gaussians(gaussian_outputs, viewpoint_camera)
                
                # Compute metrics
                l1_loss = self.l1_loss(rendered_image, gt_image)
                psnr = self.compute_psnr(rendered_image, gt_image)
                
                total_loss += l1_loss.item()
                total_psnr += psnr
                num_samples += 1
                
                # Save rendered image every 10 samples
                if batch_idx % 10 == 0:
                    self.save_rendered_image(rendered_image, batch_idx)
        
        # Log final metrics
        avg_loss = total_loss / num_samples
        avg_psnr = total_psnr / num_samples
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Average L1 Loss: {avg_loss:.6f}")
        self.logger.info(f"  Average PSNR: {avg_psnr:.2f}")
        
        return {
            'test_loss': avg_loss,
            'test_psnr': avg_psnr
        }
    
    def compute_psnr(self, img1, img2):
        """Compute PSNR between two images"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def save_rendered_image(self, image, index):
        """Save rendered image"""
        # Convert tensor to PIL Image and save
        # This is a placeholder implementation
        output_dir = Path(self.config.get('output_dir', './outputs')) / 'rendered_images'
        output_dir.mkdir(exist_ok=True)
        
        # Save logic would go here
        self.logger.info(f"Rendered image {index} saved")

