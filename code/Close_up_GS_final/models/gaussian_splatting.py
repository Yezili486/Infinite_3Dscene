"""
2D Gaussian Splatting baseline implementation
Based on 2DGS paper [10] as mentioned in Section 5.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GaussianSplatting2D(nn.Module):
    """
    2D Gaussian Splatting model implementation
    Used as baseline for Close-up-GS framework
    """
    
    def __init__(self, 
                 num_gaussians: int = 100000,
                 sh_degree: int = 3,
                 opacity_init: float = 0.1,
                 scale_init: float = 0.001):
        """
        Initialize 2DGS model
        
        Args:
            num_gaussians: Number of 2D Gaussian primitives
            sh_degree: Spherical harmonics degree for appearance
            opacity_init: Initial opacity value
            scale_init: Initial scale value
        """
        super().__init__()
        
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree + 1) ** 2
        
        # 2D Gaussian parameters
        self._xyz = nn.Parameter(torch.zeros(num_gaussians, 3))
        self._rotation = nn.Parameter(torch.zeros(num_gaussians, 4))
        self._scaling = nn.Parameter(torch.full((num_gaussians, 2), scale_init))
        self._opacity = nn.Parameter(torch.full((num_gaussians,), opacity_init))
        self._features_dc = nn.Parameter(torch.zeros(num_gaussians, 1, 3))
        self._features_rest = nn.Parameter(torch.zeros(num_gaussians, self.sh_dim - 1, 3))
        
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.rotation_activation = F.normalize
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation, dim=-1)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def forward(self, viewpoint_camera: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 2DGS rendering
        
        Args:
            viewpoint_camera: Camera parameters dictionary
            
        Returns:
            Dictionary containing rendered outputs
        """
        # This would contain the actual 2DGS rendering implementation
        # For now, returning placeholder structure
        batch_size = 1
        height, width = viewpoint_camera.get('image_height', 512), viewpoint_camera.get('image_width', 512)
        
        rendered_image = torch.zeros(batch_size, 3, height, width)
        depth_map = torch.zeros(batch_size, 1, height, width)
        alpha_map = torch.zeros(batch_size, 1, height, width)
        
        return {
            'render': rendered_image,
            'depth': depth_map,
            'alpha': alpha_map,
            'viewspace_points': self._xyz,
            'visibility_filter': torch.ones(self.num_gaussians, dtype=torch.bool),
            'radii': torch.zeros(self.num_gaussians)
        }
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, 
                            new_opacity, new_scaling, new_rotation):
        """Add new Gaussians to the model"""
        d = {
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacity,
            "_scaling": new_scaling,
            "_rotation": new_rotation
        }
        
        for key, value in d.items():
            param = getattr(self, key)
            param.data = torch.cat([param.data, value], dim=0)
        
        self.num_gaussians += new_xyz.shape[0]
    
    def prune_points(self, mask):
        """Remove Gaussians based on mask"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.num_gaussians = self._xyz.shape[0]
    
    def _prune_optimizer(self, mask):
        """Helper function for pruning optimizer tensors"""
        optimizable_tensors = {}
        optimizable_tensors["xyz"] = self._xyz[mask]
        optimizable_tensors["f_dc"] = self._features_dc[mask]
        optimizable_tensors["f_rest"] = self._features_rest[mask]
        optimizable_tensors["opacity"] = self._opacity[mask]
        optimizable_tensors["scaling"] = self._scaling[mask]
        optimizable_tensors["rotation"] = self._rotation[mask]
        return optimizable_tensors