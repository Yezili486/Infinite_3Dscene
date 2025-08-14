"""
Progressive Self-Training for Close-up-GS
Implementation of progressive update algorithm (Paper Section 4.3, Figure 2)
"""

import torch
import torch.optim as optim
import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from utils.view_selection import ViewSelector
from models.gs_model import GSModel
from models.closeup_refiner import CloseupRefiner
from utils.camera import Camera
from utils.metrics import evaluate_image_metrics

@dataclass
class ProgressiveRound:
    """Information for one progressive training round"""
    round_id: int
    scale_factor: float
    frontier_views: List[Dict]
    anchor_views: List[int]
    update_views: List[Dict]
    rendered_images: List[torch.Tensor]
    refined_images: List[torch.Tensor]
    losses: Dict[str, float]
    metrics: Dict[str, float]

class ProgressiveTrainer:
    """
    Progressive Self-Training Implementation
    Based on Paper Section 4.3, Figure 2
    """
    
    def __init__(self, 
                 gs_model: GSModel,
                 closeup_refiner: CloseupRefiner,
                 view_selector: ViewSelector,
                 config: Dict,
                 device: torch.device,
                 logger):
        """
        Initialize progressive trainer
        
        Args:
            gs_model: Gaussian Splatting model
            closeup_refiner: Close-up view refiner
            view_selector: View selection module
            config: Configuration dictionary
            device: Training device
            logger: Logger instance
        """
        self.gs_model = gs_model
        self.closeup_refiner = closeup_refiner
        self.view_selector = view_selector
        self.config = config
        self.device = device
        self.logger = logger
        
        # Progressive training parameters
        self.default_rounds = config.get('progressive_rounds', 3)
        self.default_scales = config.get('progressive_scales', [3, 9, 27])
        self.anchor_count = config.get('anchor_count', 5)
        self.update_count = config.get('update_count', 8)
        # 先从progressive_training配置中获取，然后从全局配置获取
        progressive_config = config.get('progressive_training', {})
        self.finetune_iterations = progressive_config.get('finetune_iterations', 
                                   config.get('finetune_iterations', 5))  # 默认5次用于快速验证
        
        # Distance parameters
        self.densify_threshold = config.get('densify_threshold', 1/3)
        
        # Reliable pixel parameters
        self.reliable_threshold = config.get('reliable_threshold', 0.95)
        self.geometric_consistency_threshold = config.get('geometric_consistency_threshold', 0.1)
        
        # Training history
        self.training_history = []
        
        self.logger.info(f"ProgressiveTrainer initialized")
        self.logger.info(f"  Rounds: {self.default_rounds}")
        self.logger.info(f"  Scales: {self.default_scales}")
        self.logger.info(f"  Anchor count: {self.anchor_count}")
        self.logger.info(f"  Update count: {self.update_count}")
    
    def progressive_update(self, 
                          Vs: List[Dict], 
                          p_target: torch.Tensor,
                          rounds: int = None,
                          scales: List[float] = None) -> Tuple[GSModel, List[Dict]]:
        """
        Progressive update algorithm (Paper Section 4.3, Figure 2)
        
        Args:
            Vs: Training views with cameras and images
            p_target: Object center position [3]
            rounds: Number of progressive rounds (default: 3)
            scales: Scale factors for each round (default: [3, 9, 27])
            
        Returns:
            Tuple of (updated_gs_model, updated_training_views)
        """
        if rounds is None:
            rounds = self.default_rounds
        if scales is None:
            scales = self.default_scales[:rounds]
        
        self.logger.info(f"Starting progressive update: {rounds} rounds")
        self.logger.info(f"Scales: {scales}")
        self.logger.info(f"Initial training views: {len(Vs)}")
        
        # Initialize updated views with original training views
        updated_Vs = Vs.copy()
        
        for round_idx in range(rounds):
            scale_factor = scales[round_idx] if round_idx < len(scales) else scales[-1]
            
            self.logger.info(f"\n=== Progressive Round {round_idx + 1}/{rounds} ===")
            self.logger.info(f"Scale factor: {scale_factor}")
            
            round_info = self._execute_progressive_round(
                updated_Vs, p_target, round_idx, scale_factor
            )
            
            # Update training views with newly generated views
            updated_Vs = self._update_training_views(updated_Vs, round_info)
            
            # Store round information
            self.training_history.append(round_info)
            
            self.logger.info(f"Round {round_idx + 1} completed")
            self.logger.info(f"Updated training views: {len(updated_Vs)}")
            self.logger.info(f"Round metrics: {round_info.metrics}")
        
        self.logger.info(f"Progressive update completed")
        self.logger.info(f"Final training views: {len(updated_Vs)}")
        
        return self.gs_model, updated_Vs
    
    def _execute_progressive_round(self, 
                                 Vs: List[Dict], 
                                 p_target: torch.Tensor,
                                 round_idx: int,
                                 scale_factor: float) -> ProgressiveRound:
        """Execute one progressive training round"""
        
        # Step 1: Set frontier views (α_t * d close to p_target)
        frontier_views = self._set_frontier_views(Vs, p_target, scale_factor)
        self.logger.info(f"Step 1: Set {len(frontier_views)} frontier views")
        
        # Step 2: Select anchor views (k=5)
        anchor_indices = self._select_anchor_views(Vs, frontier_views, p_target)
        self.logger.info(f"Step 2: Selected {len(anchor_indices)} anchor views: {anchor_indices}")
        
        # Step 3: Select views to be updated (M=random sampling, select 8)
        update_views = self._select_update_views(Vs, frontier_views, p_target)
        self.logger.info(f"Step 3: Selected {len(update_views)} views to update")
        
        # Step 4: Render + Refine (See3D + SUPIR)
        rendered_images, refined_images = self._render_and_refine(
            update_views, [Vs[i] for i in anchor_indices]
        )
        self.logger.info(f"Step 4: Rendered and refined {len(refined_images)} images")
        
        # Step 5: Fine-tune gs_model (reliable pixels, 5000 iterations, densify if < 1/3 dist)
        losses, metrics = self._finetune_model(
            update_views, refined_images, p_target
        )
        self.logger.info(f"Step 5: Fine-tuning completed")
        
        return ProgressiveRound(
            round_id=round_idx,
            scale_factor=scale_factor,
            frontier_views=frontier_views,
            anchor_views=anchor_indices,
            update_views=update_views,
            rendered_images=rendered_images,
            refined_images=refined_images,
            losses=losses,
            metrics=metrics
        )
    
    def _set_frontier_views(self, 
                          Vs: List[Dict], 
                          p_target: torch.Tensor, 
                          scale_factor: float) -> List[Dict]:
        """
        Set frontier views (α_t * d close to p_target)
        Paper Section 4.3: frontier views are determined by scale factor
        """
        # 步骤1: 诊断tensor设备
        if len(Vs) > 0:
            first_camera = Vs[0]['camera']
            print(f"DEBUG: camera.camera_center device: {first_camera.camera_center.device}")
            print(f"DEBUG: p_target device: {p_target.device}")
            
            # 步骤2: 移动p_target到CUDA设备
            p_target = p_target.to(first_camera.camera_center.device)
            print(f"DEBUG: p_target moved to device: {p_target.device}")
        
        frontier_views = []
        
        # Calculate base distance from training views
        distances = []
        for view in Vs:
            camera = view['camera']
            # 确保camera也在正确设备上
            camera = camera.to(p_target.device)
            distance = torch.norm(camera.camera_center - p_target)
            distances.append(distance.item())
        
        # Determine frontier distance threshold
        base_distance = np.mean(distances)
        frontier_distance = base_distance / scale_factor  # α_t * d
        
        self.logger.debug(f"Base distance: {base_distance:.3f}")
        self.logger.debug(f"Frontier distance threshold: {frontier_distance:.3f}")
        
        # Generate frontier views around p_target at frontier distance
        num_frontier_views = self.config.get('num_frontier_views', 12)
        
        for i in range(num_frontier_views):
            # Generate views in sphere around p_target
            theta = 2 * math.pi * i / num_frontier_views
            phi = math.pi / 3  # Fixed elevation angle
            
            # Position at frontier distance
            direction_vector = torch.tensor([
                math.sin(phi) * math.cos(theta),
                math.cos(phi),
                math.sin(phi) * math.sin(theta)
            ], device=p_target.device)
            pos = p_target + frontier_distance * direction_vector
            
            # Create camera looking at p_target
            forward = p_target - pos
            forward = forward / torch.norm(forward)
            
            up = torch.tensor([0., 1., 0.], device=p_target.device)
            right = torch.cross(forward, up)
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            
            # Create camera-to-world matrix
            camera_to_world = torch.eye(4, device=p_target.device)
            camera_to_world[:3, 0] = right
            camera_to_world[:3, 1] = up
            camera_to_world[:3, 2] = -forward
            camera_to_world[:3, 3] = pos
            
            # Create camera
            camera = Camera(
                image_width=self.config.get('image_width', 512),
                image_height=self.config.get('image_height', 512),
                fx=self.config.get('focal_length', 500.0),
                fy=self.config.get('focal_length', 500.0),
                cx=self.config.get('image_width', 512) / 2.0,
                cy=self.config.get('image_height', 512) / 2.0,
                world_to_camera=torch.inverse(camera_to_world),
                camera_to_world=camera_to_world,
                device=p_target.device
            )
            
            frontier_views.append({
                'camera': camera,
                'distance_to_target': frontier_distance,
                'round_generated': f"round_{len(self.training_history)}",
                'view_type': 'frontier'
            })
        
        return frontier_views
    
    def _select_anchor_views(self, 
                           Vs: List[Dict], 
                           frontier_views: List[Dict], 
                           p_target: torch.Tensor) -> List[int]:
        """Select anchor views using view selection algorithm"""
        
        # Extract poses from training views
        known_poses = []
        for view in Vs:
            known_poses.append(view['camera'].camera_to_world)
        
        # Extract poses from frontier views
        frontier_poses = []
        for view in frontier_views:
            frontier_poses.append(view['camera'].camera_to_world)
        
        # 步骤3: 确保p_target在CUDA设备上
        if len(known_poses) > 0:
            target_device = known_poses[0].device
            p_target = p_target.to(target_device)
        
        # Use view selector to select anchors
        anchor_indices, selection_info = self.view_selector.select_anchors(
            known_views=known_poses,
            frontier_views=frontier_poses,
            p_target=p_target,
            k=self.anchor_count
        )
        
        self.logger.debug(f"Anchor selection objective: {selection_info['objective_value']:.4f}")
        
        return anchor_indices
    
    def _select_update_views(self, 
                           Vs: List[Dict], 
                           frontier_views: List[Dict], 
                           p_target: torch.Tensor) -> List[Dict]:
        """
        Select views to be updated (M=random sampling, select 8)
        Paper Section 4.3: Random sampling of candidate views
        """
        
        # Random sampling from frontier views
        num_candidates = min(len(frontier_views), self.update_count * 2)  # Sample more candidates
        candidate_views = random.sample(frontier_views, num_candidates)
        
        # Extract poses for view selection algorithm
        candidate_poses = []
        for view in candidate_views:
            candidate_poses.append(view['camera'].camera_to_world)
        
        # Use anchor views as reference (first few training views as proxy)
        anchor_poses = []
        for i in range(min(self.anchor_count, len(Vs))):
            anchor_poses.append(Vs[i]['camera'].camera_to_world)
        
        # Frontier poses for selection
        frontier_poses = [view['camera'].camera_to_world for view in frontier_views]
        
        # 步骤3: 确保p_target在CUDA设备上
        if len(candidate_poses) > 0:
            target_device = candidate_poses[0].device
            p_target = p_target.to(target_device)
        
        # Select best update views
        update_indices, selection_info = self.view_selector.select_to_be_updated(
            M_samples=candidate_poses,
            anchors=anchor_poses,
            frontiers=frontier_poses,
            p_target=p_target,
            k=min(self.update_count, len(candidate_views))
        )
        
        # Return selected candidate views
        selected_views = [candidate_views[i] for i in update_indices]
        
        self.logger.debug(f"Update selection objective: {selection_info['objective_value']:.4f}")
        
        return selected_views
    
    def _render_and_refine(self, 
                         update_views: List[Dict], 
                         anchor_views: List[Dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Render + Refine step (See3D + SUPIR)
        Paper Section 4.3, Step 4
        """
        rendered_images = []
        refined_images = []
        
        self.gs_model.eval()
        
        with torch.no_grad():
            for i, update_view in enumerate(update_views):
                camera = update_view['camera']
                
                # Step 4a: Render with current GS model
                try:
                    output = self.gs_model(camera)
                    rendered_image = output['image']
                    rendered_images.append(rendered_image)
                    
                    self.logger.debug(f"Rendered view {i+1}/{len(update_views)}: {rendered_image.shape}")
                    
                except Exception as e:
                    self.logger.warning(f"Rendering failed for view {i}: {e}")
                    # Create dummy image if rendering fails
                    dummy_image = torch.zeros(3, self.config.get('image_height', 512), 
                                            self.config.get('image_width', 512), device=self.device)
                    rendered_images.append(dummy_image)
                    rendered_image = dummy_image
                
                # Step 4b: Refine with See3D + SUPIR
                try:
                    # Prepare reference views (use anchor views)
                    ref_views = []
                    for anchor_view in anchor_views[:3]:  # Use top 3 anchor views
                        ref_views.append({
                            'camera': anchor_view['camera'],
                            'image': self._render_reference_image(anchor_view['camera'])
                        })
                    
                    # Apply close-up refinement
                    refine_result = self.closeup_refiner.refine_view(camera, ref_views)
                    refined_image = refine_result['refined_image']  # Extract the actual image tensor
                    refined_images.append(refined_image)
                    
                    self.logger.debug(f"Refined view {i+1}/{len(update_views)}: {refined_image.shape}")
                    
                except Exception as e:
                    self.logger.warning(f"Refinement failed for view {i}: {e}")
                    # Use rendered image as fallback
                    refined_images.append(rendered_image)
        
        return rendered_images, refined_images
    
    def _render_reference_image(self, camera: Camera) -> torch.Tensor:
        """Render reference image for refinement"""
        try:
            with torch.no_grad():
                output = self.gs_model(camera)
                return output['image']
        except Exception as e:
            self.logger.warning(f"Reference rendering failed: {e}")
            # Return dummy image
            return torch.zeros(3, self.config.get('image_height', 512), 
                             self.config.get('image_width', 512), device=self.device)
    
    def _finetune_model(self, 
                       update_views: List[Dict], 
                       refined_images: List[torch.Tensor], 
                       p_target: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Fine-tune gs_model (reliable pixels, 5000 iterations, densify if < 1/3 dist)
        Paper Section 4.3, Step 5
        """
        
        self.logger.info(f"Fine-tuning model with {len(refined_images)} refined images")
        
        # Setup optimizer
        param_groups = [
            {'params': [self.gs_model._centers], 'lr': 0.00016, 'name': "xyz"},
            {'params': [self.gs_model._sh_coeffs], 'lr': 0.0025, 'name': "f_sh"},
            {'params': [self.gs_model._opacities], 'lr': 0.05, 'name': "opacity"},
            {'params': [self.gs_model._scales], 'lr': 0.005, 'name': "scaling"},
            {'params': [self.gs_model._rotations], 'lr': 0.001, 'name': "rotation"}
        ]
        optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # 步骤3: 确保模型在正确设备上，并清理内存
        self.gs_model = self.gs_model.to(self.device)
        self.gs_model.train()
        
        # 清理GPU内存缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        total_loss = 0.0
        losses = {'l1_loss': 0.0, 'ssim_loss': 0.0, 'total_loss': 0.0}
        
        for iteration in range(self.finetune_iterations):
            # Random sample one view for this iteration
            view_idx = random.randint(0, len(update_views) - 1)
            camera = update_views[view_idx]['camera']
            target_image = refined_images[view_idx]
            
            # Forward pass
            output = self.gs_model(camera)
            rendered_image = output['image']
            
            # Apply reliable pixel mask
            reliable_mask = self._compute_reliable_pixel_mask(
                rendered_image, target_image, camera, p_target
            )
            
            # Compute loss with reliable pixels only
            loss_dict = self._compute_masked_loss(rendered_image, target_image, reliable_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Update training stats
            if self.gs_model._centers.grad is not None:
                self.gs_model.update_training_stats(self.gs_model._centers.grad)
            
            optimizer.step()
            
            # 步骤3: 定期清理GPU内存（每100次迭代）
            if iteration % 100 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Accumulate losses
            for key in losses:
                losses[key] += loss_dict[key].item()
            
            # Densification (only when distance < 1/3)
            if (iteration >= 500 and iteration <= 3000 and iteration % 100 == 0):
                camera_distance = torch.norm(camera.camera_center - p_target)
                if camera_distance < self.densify_threshold:
                    self.gs_model.densify(camera_distance.item())
            
            # Opacity reset
            if iteration % 3000 == 0 and iteration > 0:
                self.gs_model.reset_opacity()
            
            # Pruning
            if iteration % 1000 == 0 and iteration > 0:
                self.gs_model.prune_gaussians()
            
            # Logging - 更频繁的进度显示
            if iteration % 1 == 0:  # 每次迭代都显示
                print(f"Fine-tune iter {iteration+1}/{self.finetune_iterations}: loss = {loss_dict['total_loss']:.6f}")
                self.logger.debug(f"Fine-tune iter {iteration}: loss = {loss_dict['total_loss']:.6f}")
        
        # Average losses
        for key in losses:
            losses[key] /= self.finetune_iterations
        
        # Compute final metrics
        metrics = self._compute_round_metrics(update_views, refined_images)
        
        self.logger.info(f"Fine-tuning completed: avg_loss = {losses['total_loss']:.6f}")
        
        return losses, metrics
    
    def _compute_reliable_pixel_mask(self, 
                                   rendered_image: torch.Tensor, 
                                   target_image: torch.Tensor,
                                   camera: Camera,
                                   p_target: torch.Tensor) -> torch.Tensor:
        """
        Compute reliable pixel mask using geometric consistency
        Paper: Add reliable pixels for training
        """
        H, W = rendered_image.shape[-2:]
        
        # Create base mask (all pixels initially reliable)
        mask = torch.ones(H, W, device=self.device)
        
        # Apply geometric consistency check
        # Project object center to image plane
        try:
            # 步骤2: 统一设备到CUDA，修复矩阵运算中的设备不匹配
            device = rendered_image.device
            p_target = p_target.to(device)
            camera = camera.to(device)
            
            # Simple geometric consistency: pixels near object center are more reliable
            # 确保所有tensor在同一设备上
            ones_tensor = torch.tensor([1.0], device=device)
            p_target_homo = torch.cat([p_target, ones_tensor])
            world_to_camera = camera.world_to_camera.to(device)
            
            obj_center_cam = world_to_camera @ p_target_homo
            obj_center_2d = camera.project_points(p_target.unsqueeze(0))
            
            if len(obj_center_2d) > 0:
                center_u, center_v = obj_center_2d[0]
                
                # Create distance-based reliability mask
                u_coords, v_coords = torch.meshgrid(
                    torch.arange(W, device=self.device),
                    torch.arange(H, device=self.device),
                    indexing='xy'
                )
                
                # Distance from object center
                dist_from_center = torch.sqrt((u_coords - center_u)**2 + (v_coords - center_v)**2)
                max_dist = torch.sqrt(torch.tensor(W**2 + H**2, device=self.device))
                
                # Reliability decreases with distance from center
                reliability = torch.exp(-dist_from_center / (max_dist / 3))
                mask = mask * reliability
                
        except Exception as e:
            self.logger.warning(f"Geometric consistency check failed: {e}")
        
        # Apply intensity-based reliability
        # Pixels with large intensity differences are less reliable
        intensity_diff = torch.mean(torch.abs(rendered_image - target_image), dim=0)
        intensity_mask = torch.exp(-intensity_diff / 0.5)  # Threshold of 0.5
        
        mask = mask * intensity_mask
        
        # Apply threshold
        mask = (mask > self.reliable_threshold).float()
        
        return mask
    
    def _compute_masked_loss(self, 
                           rendered_image: torch.Tensor, 
                           target_image: torch.Tensor, 
                           mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss with reliable pixel mask"""
        
        # Apply mask to images
        mask_3d = mask.unsqueeze(0).expand_as(rendered_image)
        
        masked_rendered = rendered_image * mask_3d
        masked_target = target_image * mask_3d
        
        # Compute losses only on reliable pixels
        if mask.sum() > 0:
            l1_loss = torch.nn.functional.l1_loss(masked_rendered, masked_target)
            
            # SSIM loss (simplified)
            ssim_loss = 1.0 - self._compute_ssim(masked_rendered, masked_target)
        else:
            # Fallback if no reliable pixels
            l1_loss = torch.nn.functional.l1_loss(rendered_image, target_image)
            ssim_loss = 1.0 - self._compute_ssim(rendered_image, target_image)
        
        # Combined loss
        total_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        return {
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'total_loss': total_loss
        }
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM computation"""
        # For simplicity, return a placeholder value
        # In practice, use the full SSIM implementation from models/gs_model.py
        return torch.tensor(0.8, device=img1.device)
    
    def _compute_round_metrics(self, 
                             update_views: List[Dict], 
                             refined_images: List[torch.Tensor]) -> Dict[str, float]:
        """Compute metrics for the current round"""
        
        metrics = {'psnr': 0.0, 'ssim': 0.0, 'num_views': len(update_views)}
        
        if not update_views or not refined_images:
            return metrics
        
        total_psnr = 0.0
        total_ssim = 0.0
        valid_count = 0
        
        self.gs_model.eval()
        with torch.no_grad():
            for i, (view, refined_image) in enumerate(zip(update_views, refined_images)):
                try:
                    camera = view['camera']
                    output = self.gs_model(camera)
                    rendered_image = output['image']
                    
                    # Compute metrics between rendered and refined images
                    view_metrics = evaluate_image_metrics(
                        rendered_image, refined_image, 
                        metrics=['psnr', 'ssim']
                    )
                    
                    total_psnr += view_metrics.get('psnr', 0.0)
                    total_ssim += view_metrics.get('ssim', 0.0)
                    valid_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Metrics computation failed for view {i}: {e}")
        
        if valid_count > 0:
            metrics['psnr'] = total_psnr / valid_count
            metrics['ssim'] = total_ssim / valid_count
        
        return metrics
    
    def _update_training_views(self, 
                             Vs: List[Dict], 
                             round_info: ProgressiveRound) -> List[Dict]:
        """Update training views with newly generated views"""
        
        updated_Vs = Vs.copy()
        
        # Add refined views to training set
        for i, (update_view, refined_image) in enumerate(zip(round_info.update_views, round_info.refined_images)):
            # Create new training view with refined image
            new_view = {
                'camera': update_view['camera'],
                'image': refined_image,
                'round_generated': round_info.round_id,
                'view_type': 'refined',
                'original_frontier': True
            }
            
            updated_Vs.append(new_view)
        
        self.logger.debug(f"Added {len(round_info.refined_images)} refined views to training set")
        
        return updated_Vs
    
    def get_training_history(self) -> List[ProgressiveRound]:
        """Get complete training history"""
        return self.training_history
    
    def get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics"""
        if not self.training_history:
            return {}
        
        stats = {
            'total_rounds': len(self.training_history),
            'total_refined_views': sum(len(r.refined_images) for r in self.training_history),
            'total_frontier_views': sum(len(r.frontier_views) for r in self.training_history),
            'average_psnr': np.mean([r.metrics.get('psnr', 0.0) for r in self.training_history]),
            'average_ssim': np.mean([r.metrics.get('ssim', 0.0) for r in self.training_history]),
            'scale_factors': [r.scale_factor for r in self.training_history],
            'final_loss': self.training_history[-1].losses.get('total_loss', 0.0)
        }
        
        return stats
