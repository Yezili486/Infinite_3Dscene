"""
Gaussian Splatting Model for Close-up View Synthesis
Based on Close-up-GS paper
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting Model for Close-up View Synthesis
    """
    
    def __init__(self, config):
        super(GaussianModel, self).__init__()
        self.config = config
        
        # Gaussian parameters
        self.max_gaussians = config.max_gaussians
        self.feature_dim = config.feature_dim
        
        # Initialize gaussian parameters
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, (config.sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))
        
        # Close-up enhancement modules
        self.detail_enhancer = DetailEnhancer(config)
        self.adaptive_density = AdaptiveDensity(config)
        
        # Initialize parameters
        self.setup_functions()
    
    def setup_functions(self):
        """Setup activation functions and other utilities"""
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def create_from_pcd(self, pcd, spatial_lr_scale: float = 1.0):
        """Initialize gaussians from point cloud"""
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        print(f"Number of points at initialization: {fused_point_cloud.shape[0]}")
        
        # Initialize parameters
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 
            0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        opacities = self.opacity_activation(
            torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * 0.1
        )
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            fused_color[:, None, :].contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.zeros((fused_color.shape[0], (self.config.sh_degree + 1) ** 2 - 1, 3))
            .contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, 
                            new_opacities, new_scaling, new_rotation):
        """Add new gaussians after densification"""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
    
    def forward(self, viewpoint_camera):
        """Forward pass for rendering"""
        # Get gaussian parameters
        xyz = self.get_xyz
        features = self.get_features
        scaling = self.get_scaling
        rotation = self.get_rotation
        opacity = self.get_opacity
        
        # Apply close-up enhancements
        if self.training:
            # Apply detail enhancement for close-up regions
            enhanced_features = self.detail_enhancer(features, xyz, viewpoint_camera)
            
            # Adaptive density control
            scaling, opacity = self.adaptive_density(scaling, opacity, xyz, viewpoint_camera)
        else:
            enhanced_features = features
        
        return {
            'xyz': xyz,
            'features': enhanced_features,
            'scaling': scaling,
            'rotation': rotation,
            'opacity': opacity
        }


class DetailEnhancer(nn.Module):
    """Detail enhancement module for close-up regions"""
    
    def __init__(self, config):
        super(DetailEnhancer, self).__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        
        # Detail enhancement network
        self.enhancement_net = nn.Sequential(
            nn.Linear(self.feature_dim + 3, 128),  # features + xyz
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh()
        )
    
    def forward(self, features, xyz, viewpoint_camera):
        """Enhance features for close-up regions"""
        # Compute distance to camera
        cam_pos = viewpoint_camera.camera_center
        distances = torch.norm(xyz - cam_pos, dim=1, keepdim=True)
        
        # Identify close-up regions (within threshold)
        close_mask = distances < self.config.closeup_threshold
        
        if close_mask.sum() > 0:
            # Extract features for close-up points
            close_features = features[close_mask]
            close_xyz = xyz[close_mask]
            
            # Concatenate features with position
            input_features = torch.cat([
                close_features.view(close_features.shape[0], -1),
                close_xyz
            ], dim=1)
            
            # Apply enhancement
            enhancement = self.enhancement_net(input_features)
            
            # Apply enhancement to original features
            enhanced_features = features.clone()
            enhanced_features[close_mask] = enhanced_features[close_mask] + enhancement.view_as(close_features)
            
            return enhanced_features
        
        return features


class AdaptiveDensity(nn.Module):
    """Adaptive density control for close-up regions"""
    
    def __init__(self, config):
        super(AdaptiveDensity, self).__init__()
        self.config = config
    
    def forward(self, scaling, opacity, xyz, viewpoint_camera):
        """Apply adaptive density control"""
        # Compute distance to camera
        cam_pos = viewpoint_camera.camera_center
        distances = torch.norm(xyz - cam_pos, dim=1, keepdim=True)
        
        # Adaptive scaling based on distance
        distance_factor = torch.clamp(
            1.0 / (distances / self.config.closeup_threshold + 1e-6),
            min=0.5, max=2.0
        )
        
        # Apply adaptive scaling
        adaptive_scaling = scaling * distance_factor
        
        # Adaptive opacity (higher for close-up regions)
        opacity_factor = torch.where(
            distances < self.config.closeup_threshold,
            torch.ones_like(distances) * 1.2,
            torch.ones_like(distances)
        )
        adaptive_opacity = torch.clamp(opacity * opacity_factor, max=1.0)
        
        return adaptive_scaling, adaptive_opacity


def distCUDA2(points):
    """Compute squared distances between points (placeholder)"""
    # This would typically use CUDA implementation
    # For now, use simple CPU implementation
    points_np = points.cpu().numpy()
    from scipy.spatial.distance import cdist
    distances = cdist(points_np, points_np)
    distances[distances == 0] = 1e-7  # Avoid zero distances
    min_distances = np.min(distances + np.eye(len(points_np)) * 1e10, axis=1)
    return torch.tensor(min_distances ** 2, device=points.device)

