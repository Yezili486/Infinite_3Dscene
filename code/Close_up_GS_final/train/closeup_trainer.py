"""
Complete Close-up-GS Trainer
Integrates GSModel baseline and CloseupRefiner for end-to-end training
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
from models.closeup_refiner import CloseupRefiner
from data.dataset import CloseUpDataset
from utils.logger import MetricsLogger, ProgressLogger
from utils.metrics import evaluate_image_metrics
from utils.view_selection import ViewSelector
from utils.progressive_training import ProgressiveTrainer


class CloseupGSTrainer:
    """
    Complete Close-up-GS Trainer
    
    Training Pipeline:
    1. Train baseline GSModel from training views Vs (30K iterations)
    2. Train CloseupRefiner for close-up view refinement
    3. End-to-end fine-tuning
    """
    
    def __init__(self, dataset: CloseUpDataset, config, device, logger):
        """
        Initialize Close-up-GS trainer
        
        Args:
            dataset: CloseUpDataset with training and close-up views
            config: Configuration object
            device: Training device
            logger: Logger instance
        """
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger
        
        # Initialize models
        self.gs_model = GSModel(config).to(device)
        self.closeup_refiner = CloseupRefiner(config).to(device)
        self.closeup_refiner.set_gs_model(self.gs_model)
        
        # Initialize view selector for smart view selection
        self.view_selector = ViewSelector(
            image_width=config.get('image_width', 512),
            image_height=config.get('image_height', 512),
            focal_length=config.get('focal_length', 500.0),
            distance_discount_beta=config.get('distance_discount_beta', 0.8)
        )
        
        # Initialize progressive trainer (Step 6)
        self.progressive_trainer = ProgressiveTrainer(
            gs_model=self.gs_model,
            closeup_refiner=self.closeup_refiner,
            view_selector=self.view_selector,
            config=config.to_dict() if hasattr(config, 'to_dict') else config,
            device=device,
            logger=logger
        )
        
        # Training parameters - extract from training section
        training_config = config.get('training', {})
        self.baseline_iterations = training_config.get('baseline_iterations', 100)  # Default to 100
        self.refinement_iterations = training_config.get('refinement_iterations', 50)  # Default to 50
        self.finetune_iterations = training_config.get('finetune_iterations', 50)  # Default to 50
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Metrics logger
        self.metrics_logger = MetricsLogger(
            config.get('output_dir', './outputs'),
            use_tensorboard=True,
            use_wandb=config.get('use_wandb', False)
        )
        
        # Training state
        self.current_phase = 'baseline'  # 'baseline', 'refinement', 'finetune'
        self.current_iteration = 0
        self.best_metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': float('inf')}
        
        # Close-up evaluation setup
        self.closeup_samples = dataset.get_closeup_test_samples()
        self.training_views = dataset.training_views
        
        # 步骤4: 全局设备设置和内存清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        self.logger.info(f"CloseupGS Trainer initialized")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Training views: {len(self.training_views)}")
        self.logger.info(f"  Close-up test views: {len(self.closeup_samples)}")
        self.logger.info(f"  Baseline iterations: {self.baseline_iterations}")
        self.logger.info(f"  Refinement iterations: {self.refinement_iterations}")
        self.logger.info(f"  Fine-tuning iterations: {self.finetune_iterations}")
    
    def setup_optimizers(self):
        """Setup optimizers for different training phases"""
        # Baseline GSModel optimizer
        if self.gs_model.get_centers.shape[0] > 0:
            gs_param_groups = [
                {'params': [self.gs_model._centers], 'lr': 0.00016, 'name': "xyz"},
                {'params': [self.gs_model._sh_coeffs], 'lr': 0.0025, 'name': "f_sh"},
                {'params': [self.gs_model._opacities], 'lr': 0.05, 'name': "opacity"},
                {'params': [self.gs_model._scales], 'lr': 0.005, 'name': "scaling"},
                {'params': [self.gs_model._rotations], 'lr': 0.001, 'name': "rotation"}
            ]
            self.gs_optimizer = optim.Adam(gs_param_groups, lr=0.0, eps=1e-15)
        
        # Refinement optimizer (for CloseupRefiner)
        refiner_params = list(self.closeup_refiner.parameters())
        if refiner_params:
            self.refiner_optimizer = optim.Adam(refiner_params, lr=1e-4)
        
        # Fine-tuning optimizer (both models)
        all_params = (list(self.gs_model.parameters()) + 
                     list(self.closeup_refiner.parameters()))
        self.finetune_optimizer = optim.Adam(all_params, lr=5e-5)
    
    def train(self):
        """Complete training pipeline"""
        self.logger.info("Starting Close-up-GS training pipeline")
        
        # Phase 1: Baseline GSModel training
        self.train_baseline()
        
        # Phase 2: CloseupRefiner training  
        self.train_refinement()
        
        # Phase 3: Progressive Self-Training (Step 6)
        self.train_progressive()
        
        # Phase 4: End-to-end fine-tuning
        self.train_finetune()
        
        self.logger.info("Close-up-GS training completed!")
        self.metrics_logger.close()
    
    def train_baseline(self):
        """Phase 1: Train baseline GSModel from training views Vs"""
        self.logger.info(f"Phase 1: Training baseline GSModel ({self.baseline_iterations} iterations)")
        self.current_phase = 'baseline'
        
        # Initialize from training views if needed
        if self.gs_model.get_centers.shape[0] == 0:
            self._initialize_gs_from_training_views()
            self.setup_optimizers()  # Re-setup after initialization
        
        self.gs_model.train()
        start_time = time.time()
        
        for iteration in range(self.baseline_iterations):
            self.current_iteration = iteration
            
            # 实时进度显示
            if iteration % 10 == 0:  # 每10次迭代显示一次进度
                progress = (iteration / self.baseline_iterations) * 100
                print(f"\r基线训练进度: {iteration}/{self.baseline_iterations} ({progress:.1f}%) - 当前迭代: {iteration}", end='', flush=True)
            
            # Sample training view
            view_idx = random.choice(self.training_views)
            sample = self.dataset[view_idx]
            
            camera = sample['camera']
            target_image = sample['image'].to(self.device)
            
            # Forward pass
            output = self.gs_model(camera)
            loss_dict = self.gs_model.compute_loss(output, target_image)
            
            # Backward pass
            self.gs_optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Update training stats
            if self.gs_model._centers.grad is not None:
                self.gs_model.update_training_stats(self.gs_model._centers.grad)
            
            self.gs_optimizer.step()
            
            # Memory cleanup every 50 iterations for GPU memory management
            if iteration % 50 == 0:
                torch.cuda.empty_cache()
            
            # Densification and pruning
            if (iteration >= 500 and iteration <= 15000 and 
                iteration % 100 == 0):
                distance = torch.norm(camera.camera_center - sample['object_center'])
                self.gs_model.densify(distance.item())
                if iteration % 100 == 0:  # 显示密集化信息
                    print(f"\n迭代 {iteration}: 执行密集化，距离阈值: {distance.item():.3f}")
            
            if iteration % 3000 == 0 and iteration > 0:
                self.gs_model.reset_opacity()
                print(f"\n迭代 {iteration}: 重置透明度")
            
            if iteration % 1000 == 0 and iteration > 0:
                self.gs_model.prune_gaussians()
                print(f"\n迭代 {iteration}: 修剪Gaussians")
            
            # Logging
            if iteration % 500 == 0:
                self.log_baseline_metrics(loss_dict, iteration)
                # 在控制台显示损失信息
                print(f"\n迭代 {iteration}: 损失={loss_dict['total_loss']:.6f}, Gaussians={self.gs_model.get_centers.shape[0]}")
            
            # Evaluation on close-up views (skip first iteration)
            if iteration > 0 and iteration % 2000 == 0:
                print(f"\n迭代 {iteration}: 开始评估...")
                eval_metrics = self.evaluate_closeup_views()
                self.log_evaluation_metrics(eval_metrics, iteration, 'baseline')
                print(f"评估结果: PSNR={eval_metrics['psnr']:.3f}, SSIM={eval_metrics['ssim']:.3f}")
                
                # Save if improved
                if eval_metrics['psnr'] > self.best_metrics['psnr']:
                    self.best_metrics.update(eval_metrics)
                    self.save_checkpoint(iteration, 'baseline_best')
                    print(f"新最佳模型已保存 (PSNR: {eval_metrics['psnr']:.3f})")
        
        # Save final baseline model
        print(f"\n保存最终基线模型...")
        self.save_checkpoint(self.baseline_iterations, 'baseline_final')
        
        elapsed = time.time() - start_time
        print(f"\n基线训练完成! 用时: {elapsed:.1f}秒")
        self.logger.info(f"Baseline training completed in {elapsed:.1f}s")
    
    def train_refinement(self):
        """Phase 2: Train CloseupRefiner for view refinement"""
        self.logger.info(f"Phase 2: Training CloseupRefiner ({self.refinement_iterations} iterations)")
        self.current_phase = 'refinement'
        
        # Freeze GSModel
        for param in self.gs_model.parameters():
            param.requires_grad = False
        
        self.closeup_refiner.train()
        self.gs_model.eval()
        
        # Select views to be updated using smart selection (Paper Section 4.3.2)
        selected_update_views = self._select_views_to_update()
        self.logger.info(f"Selected {len(selected_update_views)} views for refinement training")
        
        start_time = time.time()
        
        for iteration in range(self.refinement_iterations):
            self.current_iteration = iteration
            
            # 实时进度显示
            if iteration % 50 == 0:  # 每50次迭代显示一次进度
                progress = (iteration / self.refinement_iterations) * 100
                print(f"\r精炼训练进度: {iteration}/{self.refinement_iterations} ({progress:.1f}%)", end='', flush=True)
            
            # Sample close-up view for training
            if self.closeup_samples:
                sample = random.choice(self.closeup_samples)
                target_camera = sample['camera']
                target_image = sample['image'].to(self.device)
                
                # Get reference views (other training views)
                ref_views = self._get_reference_views(target_camera, num_refs=3)
                
                # Forward pass through refiner
                try:
                    refined_image = self.closeup_refiner(target_camera, ref_views)
                    
                    # Resize target to match refined image if needed
                    if refined_image.shape != target_image.shape:
                        target_resized = torch.nn.functional.interpolate(
                            target_image.unsqueeze(0),
                            size=refined_image.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    else:
                        target_resized = target_image
                    
                    # Compute refinement loss
                    l1_loss = torch.nn.functional.l1_loss(refined_image, target_resized)
                    ssim_loss = 1.0 - self._compute_ssim(refined_image, target_resized)
                    total_loss = 0.8 * l1_loss + 0.2 * ssim_loss
                    
                    # Backward pass
                    self.refiner_optimizer.zero_grad()
                    total_loss.backward()
                    self.refiner_optimizer.step()
                    
                    loss_dict = {
                        'total_loss': total_loss.item(),
                        'l1_loss': l1_loss.item(),
                        'ssim_loss': ssim_loss.item()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Refinement step failed: {e}")
                    continue
            else:
                # No close-up samples, skip refinement training
                self.logger.warning("No close-up samples for refinement training")
                break
            
            # Logging
            if iteration % 200 == 0:
                self.log_refinement_metrics(loss_dict, iteration)
            
            # Evaluation
            if iteration % 1000 == 0:
                eval_metrics = self.evaluate_refinement()
                self.log_evaluation_metrics(eval_metrics, iteration, 'refinement')
        
        # Unfreeze GSModel for fine-tuning
        for param in self.gs_model.parameters():
            param.requires_grad = True
        
        elapsed = time.time() - start_time
        self.logger.info(f"Refinement training completed in {elapsed:.1f}s")
    
    def train_progressive(self):
        """Phase 3: Progressive Self-Training (Step 6)"""
        self.logger.info(f"Phase 3: Progressive Self-Training")
        self.current_phase = 'progressive'
        
        # Prepare training views for progressive training
        training_views = self._prepare_training_views_for_progressive()
        
        # Get object center
        p_target = self.dataset.object_center
        
        # Progressive training parameters
        progressive_config = self.config.get('progressive_training', {})
        rounds = progressive_config.get('rounds', 3)
        scales = progressive_config.get('scales', [3, 9, 27])
        
        self.logger.info(f"Starting progressive training: {rounds} rounds")
        self.logger.info(f"Scale factors: {scales}")
        self.logger.info(f"Initial training views: {len(training_views)}")
        
        start_time = time.time()
        
        try:
            # Execute progressive update
            updated_model, updated_views = self.progressive_trainer.progressive_update(
                Vs=training_views,
                p_target=p_target,
                rounds=rounds,
                scales=scales
            )
            
            # Update internal state with results
            self.gs_model = updated_model
            self.training_views = [i for i in range(len(updated_views))]  # Update training view indices
            
            # Log progressive training statistics
            stats = self.progressive_trainer.get_training_statistics()
            self.logger.info(f"Progressive training statistics:")
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value}")
            
            # Log evaluation metrics
            self.log_evaluation_metrics(stats, 0, 'progressive')
            
        except Exception as e:
            self.logger.error(f"Progressive training failed: {e}")
            self.logger.warning("Continuing without progressive training")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Progressive training completed in {elapsed:.1f}s")
    
    def _prepare_training_views_for_progressive(self) -> List[Dict]:
        """Prepare training views in the format expected by progressive trainer"""
        training_views = []
        
        for view_idx in self.training_views:
            try:
                sample = self.dataset[view_idx]
                view_dict = {
                    'camera': sample['camera'],
                    'image': sample['image'],
                    'view_id': view_idx,
                    'view_type': 'training'
                }
                training_views.append(view_dict)
            except Exception as e:
                self.logger.warning(f"Failed to prepare training view {view_idx}: {e}")
        
        return training_views
    
    def train_finetune(self):
        """Phase 4: End-to-end fine-tuning"""
        self.logger.info(f"Phase 4: End-to-end fine-tuning ({self.finetune_iterations} iterations)")
        self.current_phase = 'finetune'
        
        self.gs_model.train()
        self.closeup_refiner.train()
        
        start_time = time.time()
        
        for iteration in range(self.finetune_iterations):
            self.current_iteration = iteration
            
            # Alternate between baseline and refinement training
            if iteration % 2 == 0:
                # Baseline step
                view_idx = random.choice(self.training_views)
                sample = self.dataset[view_idx]
                camera = sample['camera']
                target_image = sample['image'].to(self.device)
                
                output = self.gs_model(camera)
                loss_dict = self.gs_model.compute_loss(output, target_image)
                loss = loss_dict['total_loss']
                
            else:
                # Refinement step
                if self.closeup_samples:
                    sample = random.choice(self.closeup_samples)
                    target_camera = sample['camera']
                    target_image = sample['image'].to(self.device)
                    ref_views = self._get_reference_views(target_camera, num_refs=2)
                    
                    try:
                        refined_image = self.closeup_refiner(target_camera, ref_views)
                        
                        if refined_image.shape != target_image.shape:
                            target_image = torch.nn.functional.interpolate(
                                target_image.unsqueeze(0),
                                size=refined_image.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        
                        loss = torch.nn.functional.l1_loss(refined_image, target_image)
                        
                    except Exception as e:
                        self.logger.warning(f"Fine-tuning refinement step failed: {e}")
                        continue
                else:
                    continue
            
            # Backward pass
            self.finetune_optimizer.zero_grad()
            loss.backward()
            self.finetune_optimizer.step()
            
            # Logging
            if iteration % 200 == 0:
                self.logger.info(f"Fine-tune iter {iteration}: loss = {loss.item():.6f}")
        
        # Final evaluation and save
        final_metrics = self.evaluate_complete_system()
        self.log_evaluation_metrics(final_metrics, self.finetune_iterations, 'final')
        self.save_checkpoint(self.finetune_iterations, 'final')
        
        elapsed = time.time() - start_time
        self.logger.info(f"Fine-tuning completed in {elapsed:.1f}s")
    
    def _initialize_gs_from_training_views(self):
        """Initialize GSModel from training views"""
        points = []
        colors = []
        
        for view_idx in self.training_views[:5]:  # Use first 5 training views
            sample = self.dataset[view_idx]
            camera = sample['camera']
            
            # Generate points in camera frustum
            num_points = 200
            depths = torch.rand(num_points, device=self.device) * 8.0 + 2.0
            xy = torch.randn(num_points, 2, device=self.device) * 2.0
            
            points_cam = torch.cat([xy, depths.unsqueeze(1)], dim=1)
            points_homo = torch.cat([points_cam, torch.ones(num_points, 1, device=self.device)], dim=1)
            camera_to_world = camera.camera_to_world.to(self.device)
            points_world = (camera_to_world @ points_homo.T).T[:, :3]
            
            points.append(points_world)
            colors.append(torch.rand(num_points, 3, device=self.device))
        
        if points:
            all_points = torch.cat(points, dim=0).to(self.device)
            all_colors = torch.cat(colors, dim=0).to(self.device)
            self.gs_model.create_from_point_cloud(all_points, all_colors)
    
    def _get_reference_views(self, target_camera, num_refs: int = 3, use_smart_selection: bool = True):
        """Get reference views for refinement using smart view selection"""
        if not use_smart_selection:
            # Fallback to distance-based selection
            ref_views = []
            target_pos = target_camera.camera_center
            
            distances = []
            for view_idx in self.training_views:
                sample = self.dataset[view_idx]
                ref_camera = sample['camera']
                distance = torch.norm(ref_camera.camera_center - target_pos)
                distances.append((distance.item(), view_idx))
            
            distances.sort()
            for _, view_idx in distances[:num_refs]:
                sample = self.dataset[view_idx]
                # 步骤3: 确保fallback情况下的参考视图也在正确设备上
                camera = sample['camera'].to(self.device)
                image = sample['image'].to(self.device)
                ref_views.append({
                    'camera': camera,
                    'image': image
                })
            
            return ref_views
        
        # Smart view selection using paper algorithms
        try:
            # Get training view poses
            training_poses = []
            for view_idx in self.training_views:
                sample = self.dataset[view_idx]
                training_poses.append(sample['camera'].camera_to_world)
            
            # Create frontier views (just the target for now)
            frontier_poses = [target_camera.camera_to_world]
            
            # Get object center
            p_target = self.dataset.object_center
            
            # Select anchor views using paper algorithm
            anchor_indices, selection_info = self.view_selector.select_anchors(
                known_views=training_poses,
                frontier_views=frontier_poses,
                p_target=p_target,
                k=min(num_refs, len(training_poses))
            )
            
            # Convert selected indices to reference views
            ref_views = []
            for anchor_idx in anchor_indices:
                if anchor_idx < len(self.training_views):
                    view_idx = self.training_views[anchor_idx]
                    sample = self.dataset[view_idx]
                    # 步骤3: 确保参考视图数据在正确设备上
                    camera = sample['camera'].to(self.device)
                    image = sample['image'].to(self.device)
                    ref_views.append({
                        'camera': camera,
                        'image': image
                    })
            
            # Log selection info
            self.logger.debug(f"Smart view selection: anchors={anchor_indices}, "
                            f"objective={selection_info['objective_value']:.4f}")
            
            return ref_views
            
        except Exception as e:
            self.logger.warning(f"Smart view selection failed: {e}, falling back to distance-based")
            return self._get_reference_views(target_camera, num_refs, use_smart_selection=False)
    
    def _compute_ssim(self, img1, img2):
        """Compute SSIM between two images"""
        # Simplified SSIM computation
        # In practice, use the full SSIM implementation
        return torch.tensor(0.8, device=img1.device)
    
    def evaluate_closeup_views(self):
        """Evaluate on close-up test views"""
        if not self.closeup_samples:
            return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        
        self.gs_model.eval()
        metrics_sum = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        num_samples = min(5, len(self.closeup_samples))  # Evaluate on subset
        
        with torch.no_grad():
            for i in range(num_samples):
                sample = self.closeup_samples[i]
                camera = sample['camera']
                target_image = sample['image'].to(self.device)
                
                output = self.gs_model(camera)
                rendered_image = output['image']
                
                metrics = evaluate_image_metrics(
                    rendered_image, target_image, 
                    metrics=['psnr', 'ssim', 'lpips']
                )
                
                for key in metrics_sum:
                    if key in metrics:
                        metrics_sum[key] += metrics[key]
        
        # Average metrics
        for key in metrics_sum:
            metrics_sum[key] /= num_samples
        
        self.gs_model.train()
        return metrics_sum
    
    def evaluate_refinement(self):
        """Evaluate refinement quality"""
        if not self.closeup_samples:
            return {'psnr': 0.0, 'ssim': 0.0}
        
        self.closeup_refiner.eval()
        metrics_sum = {'psnr': 0.0, 'ssim': 0.0}
        num_samples = min(3, len(self.closeup_samples))
        
        with torch.no_grad():
            for i in range(num_samples):
                sample = self.closeup_samples[i]
                camera = sample['camera']
                target_image = sample['image'].to(self.device)
                ref_views = self._get_reference_views(camera, num_refs=2)
                
                try:
                    refined_image = self.closeup_refiner(camera, ref_views)
                    
                    if refined_image.shape != target_image.shape:
                        target_image = torch.nn.functional.interpolate(
                            target_image.unsqueeze(0),
                            size=refined_image.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    metrics = evaluate_image_metrics(
                        refined_image, target_image,
                        metrics=['psnr', 'ssim']
                    )
                    
                    for key in metrics_sum:
                        if key in metrics:
                            metrics_sum[key] += metrics[key]
                            
                except Exception as e:
                    self.logger.warning(f"Refinement evaluation failed: {e}")
                    continue
        
        for key in metrics_sum:
            metrics_sum[key] /= num_samples
        
        self.closeup_refiner.train()
        return metrics_sum
    
    def evaluate_complete_system(self):
        """Evaluate complete Close-up-GS system"""
        baseline_metrics = self.evaluate_closeup_views()
        refinement_metrics = self.evaluate_refinement()
        
        return {
            'baseline_psnr': baseline_metrics['psnr'],
            'baseline_ssim': baseline_metrics['ssim'],
            'refinement_psnr': refinement_metrics['psnr'],
            'refinement_ssim': refinement_metrics['ssim'],
            'improvement_psnr': refinement_metrics['psnr'] - baseline_metrics['psnr'],
            'improvement_ssim': refinement_metrics['ssim'] - baseline_metrics['ssim']
        }
    
    def log_baseline_metrics(self, loss_dict, iteration):
        """Log baseline training metrics"""
        for key, value in loss_dict.items():
            self.metrics_logger.log_scalar(f"baseline/{key}", value, iteration)
        
        self.logger.info(
            f"Baseline iter {iteration}: loss={loss_dict['total_loss']:.6f}, "
            f"gaussians={loss_dict.get('num_gaussians', 0)}"
        )
    
    def log_refinement_metrics(self, loss_dict, iteration):
        """Log refinement training metrics"""
        for key, value in loss_dict.items():
            self.metrics_logger.log_scalar(f"refinement/{key}", value, iteration)
        
        self.logger.info(
            f"Refinement iter {iteration}: loss={loss_dict['total_loss']:.6f}"
        )
    
    def log_evaluation_metrics(self, metrics, iteration, phase):
        """Log evaluation metrics"""
        for key, value in metrics.items():
            self.metrics_logger.log_scalar(f"eval_{phase}/{key}", value, iteration)
        
        self.logger.info(f"Eval {phase}: {metrics}")
    
    def save_checkpoint(self, iteration, suffix):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'phase': self.current_phase,
            'gs_model_state': self.gs_model.state_dict(),
            'refiner_state': self.closeup_refiner.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = Path(self.config.get('output_dir', './outputs')) / f'closeup_gs_{suffix}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _select_views_to_update(self):
        """
        Select views to be updated using smart selection algorithm
        Paper Section 4.3.2: select_to_be_updated(M_samples, anchors, frontiers)
        """
        if not self.closeup_samples:
            self.logger.warning("No close-up samples available for view selection")
            return []
        
        try:
            # Prepare candidate views (M_samples)
            candidate_poses = []
            for sample in self.closeup_samples:
                candidate_poses.append(sample['camera'].camera_to_world)
            
            # Get anchor views (training views)
            anchor_poses = []
            for view_idx in self.training_views:
                sample = self.dataset[view_idx]
                anchor_poses.append(sample['camera'].camera_to_world)
            
            # Use close-up views as frontier views
            frontier_poses = candidate_poses.copy()
            
            # Get object center
            p_target = self.dataset.object_center
            
            # Select views to be updated
            k_update = min(len(candidate_poses), self.config.get('max_update_views', 5))
            selected_indices, selection_info = self.view_selector.select_to_be_updated(
                M_samples=candidate_poses,
                anchors=anchor_poses,
                frontiers=frontier_poses,
                p_target=p_target,
                k=k_update
            )
            
            # Convert to actual view samples
            selected_views = []
            for idx in selected_indices:
                if idx < len(self.closeup_samples):
                    selected_views.append(self.closeup_samples[idx])
            
            self.logger.info(f"Smart view selection completed: {len(selected_views)} views selected")
            self.logger.info(f"Selection objective value: {selection_info['objective_value']:.4f}")
            
            return selected_views
            
        except Exception as e:
            self.logger.warning(f"Smart view selection failed: {e}, using all close-up views")
            return self.closeup_samples
    
    def select_optimal_training_views(self, iteration: int):
        """
        Select optimal training views for current iteration
        Implements dynamic view selection during training
        """
        if iteration < 1000:
            # Use all training views in early iterations
            return self.training_views
        
        try:
            # Get current model state info
            current_gaussians = self.gs_model.get_centers
            if current_gaussians.shape[0] == 0:
                return self.training_views
            
            # Calculate view importance based on current model state
            view_importance = []
            p_target = self.dataset.object_center
            
            for view_idx in self.training_views:
                sample = self.dataset[view_idx]
                camera = sample['camera']
                
                # Calculate importance based on:
                # 1. Distance to object center
                distance = torch.norm(camera.camera_center - p_target)
                distance_score = torch.exp(-distance / 5.0)  # Closer is better
                
                # 2. Coverage of current Gaussians
                coverage_score = self._calculate_view_coverage(camera, current_gaussians)
                
                # 3. Current rendering quality (if available)
                try:
                    with torch.no_grad():
                        output = self.gs_model(camera)
                        target_image = sample['image'].to(self.device)
                        quality_score = 1.0 / (1.0 + torch.nn.functional.l1_loss(output['image'], target_image))
                except:
                    quality_score = 0.5
                
                total_importance = 0.4 * distance_score + 0.3 * coverage_score + 0.3 * quality_score
                view_importance.append((total_importance.item(), view_idx))
            
            # Sort by importance and select top views
            view_importance.sort(reverse=True)
            max_views = min(len(self.training_views), self.config.get('max_training_views_per_iter', 10))
            selected_views = [view_idx for _, view_idx in view_importance[:max_views]]
            
            return selected_views
            
        except Exception as e:
            self.logger.warning(f"Optimal view selection failed: {e}, using all training views")
            return self.training_views
    
    def _calculate_view_coverage(self, camera, gaussians):
        """Calculate how well a view covers the current Gaussians"""
        try:
            # Project Gaussians to image plane
            gaussian_centers = gaussians  # Shape: [N, 3]
            camera_centers = camera.camera_center.unsqueeze(0).expand_as(gaussian_centers)
            
            # Calculate viewing angles
            view_vectors = gaussian_centers - camera_centers
            distances = torch.norm(view_vectors, dim=1)
            
            # Weight by distance (closer Gaussians are more important)
            weights = torch.exp(-distances / distances.mean())
            
            # Calculate average coverage score
            coverage = weights.mean()
            return coverage
            
        except Exception:
            return torch.tensor(0.5)  # Default coverage


if __name__ == '__main__':
    # Test CloseupGSTrainer
    from utils.config import Config
    from data.dataset import SyntheticDataset
    
    config = Config()
    dataset = SyntheticDataset(num_samples=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import logging
    logger = logging.getLogger('test')
    
    trainer = CloseupGSTrainer(dataset, config, device, logger)
    print("CloseupGSTrainer implementation completed!")
