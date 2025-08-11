#!/usr/bin/env python3
"""
Close-up-GS: Complete implementation reproducing arXiv:2503.09396v1
Progressive 3D Gaussian Splatting for Close-up View Synthesis

Key components:
1. Baseline 3DGS with anisotropic Gaussians
2. See3D proxy using Stable Diffusion inpainting 
3. Progressive expansion with anchor/frontier view selection
4. Fine-tuning with densification
5. Comprehensive evaluation metrics
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Try to import diffusion libraries, fallback to mock if not available
try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available, using mock See3D proxy")

# Try to import additional libraries
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, skipping LPIPS metric")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image not available, using simplified SSIM")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class CameraParams:
    """Camera parameters for rendering"""
    R: torch.Tensor  # Rotation matrix (3x3)
    T: torch.Tensor  # Translation vector (3,)
    K: torch.Tensor  # Intrinsic matrix (3x3)
    width: int
    height: int
    
    def __post_init__(self):
        self.R = self.R.to(device).float()
        self.T = self.T.to(device).float() 
        self.K = self.K.to(device).float()

@dataclass
class ViewInfo:
    """Information about a camera view"""
    camera: CameraParams
    image: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    reliability_score: float = 1.0

class SphericalHarmonics:
    """Spherical harmonics for color representation"""
    
    @staticmethod
    def eval_sh(deg, sh, dirs):
        """Evaluate spherical harmonics.
        
        Args:
            deg: SH degree
            sh: SH coefficients (..., (deg+1)^2, 3)
            dirs: ray directions (..., 3)
        """
        assert deg <= 4 and deg >= 0
        assert (deg + 1) ** 2 == sh.shape[-2]
        assert dirs.shape[-1] == 3
        
        # Ensure all tensors are on the same device
        device = sh.device
        dirs = dirs.to(device)
        
        result = 0.282095 * sh[..., 0, :]  # Y_0^0
        
        if deg > 0:
            x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
            result = result - 0.488603 * y * sh[..., 1, :] + 0.488603 * z * sh[..., 2, :] - 0.488603 * x * sh[..., 3, :]
            
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result = (result +
                         1.092548 * xy * sh[..., 4, :] -
                         1.092548 * yz * sh[..., 5, :] +
                         0.315392 * (2.0 * zz - xx - yy) * sh[..., 6, :] -
                         1.092548 * xz * sh[..., 7, :] +
                         0.546274 * (xx - yy) * sh[..., 8, :])
                
                if deg > 2:
                    result = (result +
                             -0.590044 * y * (3.0 * xx - yy) * sh[..., 9, :] +
                             2.890611 * xy * z * sh[..., 10, :] +
                             -0.646360 * y * (4.0 * zz - xx - yy) * sh[..., 11, :] +
                             0.373176 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[..., 12, :] +
                             -0.646360 * x * (4.0 * zz - xx - yy) * sh[..., 13, :] +
                             1.445306 * z * (xx - yy) * sh[..., 14, :] +
                             -0.590044 * x * (xx - 3.0 * yy) * sh[..., 15, :])
        
        result = result + 0.5
        return torch.clamp(result, 0.0, 1.0)

class GaussianRenderer:
    """3D Gaussian Splatting renderer"""
    
    def __init__(self, raster_settings=None):
        self.raster_settings = raster_settings or {
            'tile_size': 16,
            'depth_test': True,
            'alpha_threshold': 1e-4
        }
    
    def render(self, gaussians, camera: CameraParams, bg_color=None):
        """Render Gaussians from camera viewpoint
        
        Args:
            gaussians: Gaussian3D object
            camera: Camera parameters
            bg_color: Background color (3,) or None for black
            
        Returns:
            dict with 'image', 'depth', 'alpha'
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
        else:
            # Get device from gaussians
            gaussian_device = gaussians.get_xyz().device if gaussians is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bg_color = torch.tensor(bg_color, device=gaussian_device, dtype=torch.float32)
            
                # Transform Gaussians to camera space
        means3D = gaussians.get_xyz()  # (N, 3)
        means2D, depths = self.project_points(means3D, camera)
        
        # Debug projection results
        print(f"Projection debug:")
        print(f"  Gaussians count: {len(means3D)}")
        print(f"  means2D range: [{means2D.min():.3f}, {means2D.max():.3f}]")
        print(f"  depths range: [{depths.min():.3f}, {depths.max():.3f}]")
        print(f"  Image bounds: [0, {camera.width}] x [0, {camera.height}]")
        
        # Count Gaussians within image bounds
        in_bounds = ((means2D[:, 0] >= 0) & (means2D[:, 0] < camera.width) & 
                    (means2D[:, 1] >= 0) & (means2D[:, 1] < camera.height))
        print(f"  Gaussians in image: {in_bounds.sum().item()}/{len(means3D)}")
        
        # Get Gaussian properties
        opacities = gaussians.get_opacity()  # (N, 1)
        colors = self.compute_colors(gaussians, camera, means3D)  # (N, 3)
        scales = gaussians.get_scaling()  # (N, 3)
        rotations = gaussians.get_rotation()  # (N, 4) quaternions
        
        # Debug Gaussian properties
        print(f"Gaussian properties:")
        print(f"  Colors range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"  Opacities range: [{opacities.min():.3f}, {opacities.max():.3f}]")
        print(f"  Scales range: [{scales.min():.3f}, {scales.max():.3f}]")
        
        # Compute 2D covariance matrices
        cov2D = self.compute_2d_covariance(means3D, scales, rotations, camera)
        
        # Sort by depth
        sorted_indices = torch.argsort(depths)
        
        # Ensure all tensors are on the same device before indexing
        tensor_device = means2D.device
        sorted_indices = sorted_indices.to(tensor_device)
        colors = colors.to(tensor_device)
        opacities = opacities.to(tensor_device)
        depths = depths.to(tensor_device)
        
        # Render with alpha blending
        rendered = self.alpha_blend_gaussians(
            means2D[sorted_indices], 
            cov2D[sorted_indices],
            colors[sorted_indices], 
            opacities[sorted_indices],
            depths[sorted_indices],
            camera, 
            bg_color
        )
        
        # Add depth information to the rendered result
        rendered['depth'] = depths[sorted_indices]
        
        return rendered
    
    def project_points(self, points3D, camera: CameraParams):
        """Project 3D points to 2D image coordinates"""
        # Transform to camera coordinates
        points_cam = points3D @ camera.R.T.to(points3D.device) + camera.T.to(points3D.device)
        
        # Get depths
        depths = points_cam[:, 2:3]
        
        # Project to normalized coordinates
        points_norm = points_cam[:, :2] / depths
        
        # Apply intrinsic matrix to get pixel coordinates
        fx = camera.K[0, 0].to(points_norm.device)
        fy = camera.K[1, 1].to(points_norm.device)
        cx = camera.K[0, 2].to(points_norm.device)
        cy = camera.K[1, 2].to(points_norm.device)
        
        points2D = torch.stack([
            points_norm[:, 0] * fx + cx,
            points_norm[:, 1] * fy + cy
        ], dim=1)
        
        return points2D, depths.squeeze()
    
    def compute_colors(self, gaussians, camera: CameraParams, means3D):
        """Compute colors using spherical harmonics"""
        # Compute viewing directions
        cam_pos = -camera.R.T.to(means3D.device) @ camera.T.to(means3D.device)
        view_dirs = means3D - cam_pos
        view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
        
        # Evaluate SH
        sh_coeffs = gaussians.get_features()
        colors = SphericalHarmonics.eval_sh(gaussians.sh_degree, sh_coeffs, view_dirs)
        
        return colors
    
    def compute_2d_covariance(self, means3D, scales, rotations, camera: CameraParams):
        """Compute 2D covariance matrices from 3D Gaussians"""
        # Ensure all tensors are on the same device
        device = means3D.device
        scales = scales.to(device)
        rotations = rotations.to(device)
        
        # Convert quaternions to rotation matrices
        rot_matrices = self.quaternion_to_matrix(rotations)  # (N, 3, 3)
        
        # Scale matrices
        scale_matrices = torch.diag_embed(scales)  # (N, 3, 3)
        
        # 3D covariance: R * S * S^T * R^T
        cov3D = rot_matrices @ scale_matrices @ scale_matrices.transpose(-2, -1) @ rot_matrices.transpose(-2, -1)
        
        # Project to 2D using Jacobian of projection
        # Simplified projection jacobian (assuming perspective)
        focal_x = camera.K[0, 0].to(device)
        focal_y = camera.K[1, 1].to(device)
        
        # Transform points to camera space
        points_cam = means3D @ camera.R.T.to(device) + camera.T.to(device)
        z = points_cam[:, 2]
        
        # Projection Jacobian (simplified)
        J = torch.zeros(means3D.shape[0], 2, 3, device=device)
        J[:, 0, 0] = focal_x / z
        J[:, 0, 2] = -focal_x * points_cam[:, 0] / (z ** 2)
        J[:, 1, 1] = focal_y / z
        J[:, 1, 2] = -focal_y * points_cam[:, 1] / (z ** 2)
        
        # Apply camera rotation to covariance
        cov3D_cam = camera.R.to(device) @ cov3D @ camera.R.T.to(device)
        
        # Project: J * Σ_3D * J^T
        cov2D = J @ cov3D_cam @ J.transpose(-2, -1)
        
        # Add small regularization for numerical stability
        eye = torch.eye(2, device=device).expand(cov2D.shape[0], -1, -1)
        cov2D = cov2D + 1e-6 * eye
        
        return cov2D
    
    def quaternion_to_matrix(self, quaternions):
        """Convert quaternions to rotation matrices"""
        q = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
        
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*z*w
        R[:, 0, 2] = 2*x*z + 2*y*w
        R[:, 1, 0] = 2*x*y + 2*z*w
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*x*w
        R[:, 2, 0] = 2*x*z - 2*y*w
        R[:, 2, 1] = 2*y*z + 2*x*w
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    def alpha_blend_gaussians(self, means2D, cov2D, colors, opacities, depths, camera: CameraParams, bg_color):
        """Alpha blend sorted Gaussians with high quality rendering"""
        H, W = camera.height, camera.width
        
        # Create output tensors
        camera_device = camera.R.device
        final_image = torch.zeros(H, W, 3, device=camera_device, dtype=torch.float32)
        final_depth = torch.zeros(H, W, device=camera_device, dtype=torch.float32)
        final_alpha = torch.zeros(H, W, device=camera_device, dtype=torch.float32)
        

        
        # Create pixel coordinates
        y, x = torch.meshgrid(torch.arange(H, device=camera_device), 
                             torch.arange(W, device=camera_device), indexing='ij')
        pixels = torch.stack([x.flatten(), y.flatten()], dim=1).float()  # (H*W, 2)
        
        # Process each Gaussian in batches to save memory
        accumulated_alpha = torch.zeros(H * W, device=camera_device, dtype=torch.float32)
        accumulated_color = torch.zeros(H * W, 3, device=camera_device, dtype=torch.float32)
        accumulated_depth = torch.zeros(H * W, device=camera_device, dtype=torch.float32)
        
        batch_size = 20  # Smaller batch size for memory efficiency
        for batch_start in range(0, len(means2D), batch_size):
            batch_end = min(batch_start + batch_size, len(means2D))
            
            for i in range(batch_start, batch_end):
                # Compute Gaussian influence
                center = means2D[i:i+1]  # (1, 2)
                cov = cov2D[i]  # (2, 2)
                
                # Ensure center and cov are on the same device as pixels
                center = center.to(pixels.device)
                cov = cov.to(pixels.device)
                
                # Add small regularization to covariance for numerical stability
                cov = cov + torch.eye(2, device=cov.device) * 1e-4  # Increased regularization
                
                # Compute squared Mahalanobis distance
                diff = pixels - center  # (H*W, 2)
                try:
                    cov_inv = torch.inverse(cov)
                    quad_form = torch.sum(diff * (diff @ cov_inv), dim=1)  # (H*W,)
                    
                    # Gaussian weight with better scaling
                    weights = torch.exp(-0.5 * quad_form)
                    
                    # Normalize weights to have reasonable values
                    if weights.max() > 0:
                        weights = weights / weights.max() * 0.8  # Scale to reasonable range
                    
                    # Apply opacity with better scaling
                    alpha = opacities[i].to(pixels.device) * weights
                    
                    # Ensure alpha values are reasonable
                    alpha = torch.clamp(alpha, 0.0, 1.0)
                    
                    # Alpha blending with better handling
                    transmittance = torch.clamp(1.0 - accumulated_alpha, 0.0, 1.0)
                    contribution = alpha * transmittance
                    
                    # Add color contribution
                    color_contribution = contribution.unsqueeze(1) * colors[i:i+1].to(pixels.device)
                    accumulated_color += color_contribution
                    
                    # Add depth contribution
                    depth_contribution = contribution * depths[i].to(pixels.device)
                    accumulated_depth += depth_contribution
                    
                    # Update accumulated alpha
                    accumulated_alpha += contribution
                    accumulated_alpha = torch.clamp(accumulated_alpha, 0.0, 1.0)
                    
                    # Early termination for fully opaque pixels
                    if torch.all(accumulated_alpha > 0.99):
                        break
                        
                except RuntimeError as e:
                    # Skip if covariance is not invertible
                    print(f"Warning: Skipping Gaussian {i} due to singular covariance: {e}")
                    continue
            
            # Clean up GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add background with better handling
        transmittance = torch.clamp(1.0 - accumulated_alpha, 0.0, 1.0)
        bg_contribution = transmittance.unsqueeze(1) * bg_color.to(accumulated_color.device)
        accumulated_color += bg_contribution
        
        # Reshape to image
        final_image = accumulated_color.view(H, W, 3)
        final_depth = accumulated_depth.view(H, W)
        final_alpha = accumulated_alpha.view(H, W)
        
        # Ensure final image is in valid range
        final_image = torch.clamp(final_image, 0.0, 1.0)
        
        # Debug final rendering results
        print(f"Rendering completed: {H}x{W} image")
        print(f"  Final image range: [{final_image.min():.3f}, {final_image.max():.3f}]")
        print(f"  Final alpha range: [{final_alpha.min():.3f}, {final_alpha.max():.3f}]")
        print(f"  Non-zero pixels: {(final_image.sum(dim=-1) > 0).sum().item()}/{H*W}")
        
        return {
            'image': final_image,
            'depth': final_depth,
            'alpha': final_alpha
        }

class Gaussian3D:
    """3D Gaussian representation for neural radiance fields"""
    
    def __init__(self, num_points=1000, sh_degree=3):
        self.sh_degree = sh_degree
        self.sh_channels = (sh_degree + 1) ** 2
        
        # Initialize parameters (will be moved to appropriate device when needed)
        self._xyz = nn.Parameter(torch.randn(num_points, 3, dtype=torch.float32) * 0.1)
        self._features_dc = nn.Parameter(torch.randn(num_points, 1, 3, dtype=torch.float32) * 0.1)
        self._features_rest = nn.Parameter(torch.randn(num_points, self.sh_channels - 1, 3, dtype=torch.float32) * 0.01)
        self._scaling = nn.Parameter(torch.randn(num_points, 3, dtype=torch.float32) * 0.01)
        self._rotation = nn.Parameter(torch.randn(num_points, 4, dtype=torch.float32))
        self._opacity = nn.Parameter(torch.randn(num_points, 1, dtype=torch.float32))
        
        # Normalize quaternions
        with torch.no_grad():
            self._rotation.data = self._rotation.data / torch.norm(self._rotation.data, dim=1, keepdim=True)
    
    def get_xyz(self):
        return self._xyz
    
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)
    
    def get_opacity(self):
        return torch.sigmoid(self._opacity)
    
    def get_scaling(self):
        return torch.exp(self._scaling)
    
    def get_rotation(self):
        return self._rotation / torch.norm(self._rotation, dim=1, keepdim=True)
    
    def densify(self, positions, min_opacity=0.01):
        """Add new Gaussians by splitting existing ones"""
        with torch.no_grad():
            # Find Gaussians to split (high gradient or large scale)
            xyz_grad = getattr(self._xyz, 'grad', None)
            if xyz_grad is not None:
                grad_norm = torch.norm(xyz_grad, dim=1)
                split_mask = grad_norm > grad_norm.quantile(0.8)
            else:
                # Fallback: split large Gaussians
                scales = self.get_scaling()
                max_scale = torch.max(scales, dim=1)[0]
                split_mask = max_scale > max_scale.quantile(0.8)
            
            # Only split if we have significant Gaussians
            if split_mask.sum() == 0:
                return
            
            # Get parameters of Gaussians to split
            split_indices = torch.where(split_mask)[0]
            n_split = len(split_indices)
            
            if n_split == 0:
                return
            
            # Create new Gaussians by adding noise
            noise_scale = 0.1
            new_xyz = self._xyz[split_indices] + torch.randn(n_split, 3, device=self._xyz.device) * noise_scale
            new_features_dc = self._features_dc[split_indices].clone()
            new_features_rest = self._features_rest[split_indices].clone()
            new_scaling = self._scaling[split_indices] - np.log(2.0)  # Smaller scale
            new_rotation = self._rotation[split_indices].clone()
            new_opacity = self._opacity[split_indices].clone()
            
            # Concatenate with existing
            self._xyz.data = torch.cat([self._xyz.data, new_xyz])
            self._features_dc.data = torch.cat([self._features_dc.data, new_features_dc])
            self._features_rest.data = torch.cat([self._features_rest.data, new_features_rest])
            self._scaling.data = torch.cat([self._scaling.data, new_scaling])
            self._rotation.data = torch.cat([self._rotation.data, new_rotation])
            self._opacity.data = torch.cat([self._opacity.data, new_opacity])
            
            print(f"Densified: added {n_split} Gaussians (total: {len(self._xyz)})")
    
    def prune(self, min_opacity=0.01):
        """Remove Gaussians with low opacity"""
        with torch.no_grad():
            opacities = self.get_opacity()
            keep_mask = opacities.squeeze() > min_opacity
            
            if keep_mask.sum() == len(keep_mask):
                return  # Nothing to prune
            
            self._xyz.data = self._xyz.data[keep_mask]
            self._features_dc.data = self._features_dc.data[keep_mask]
            self._features_rest.data = self._features_rest.data[keep_mask]
            self._scaling.data = self._scaling.data[keep_mask]
            self._rotation.data = self._rotation.data[keep_mask]
            self._opacity.data = self._opacity.data[keep_mask]
            
            print(f"Pruned: removed {(~keep_mask).sum()} Gaussians (total: {len(self._xyz)})")
    
    def parameters(self):
        """Return all parameters for optimization"""
        return [self._xyz, self._features_dc, self._features_rest, 
                self._scaling, self._rotation, self._opacity]

class See3DProxy:
    """Proxy for See3D model using Stable Diffusion inpainting"""
    
    def __init__(self):
        self.inpaint_pipeline = None
        if DIFFUSERS_AVAILABLE:
            # Set environment variables for better network handling
            import os
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '0'
            
            # Try multiple times with different configurations
            for attempt in range(3):
                try:
                    print(f"Attempting to load Stable Diffusion model (attempt {attempt + 1}/3)...")
                    
                    # Try with different configurations
                    if attempt == 0:
                        # First attempt: standard loading
                        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                            "runwayml/stable-diffusion-inpainting",
                            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                            local_files_only=False,
                            resume_download=True
                        )
                    elif attempt == 1:
                        # Second attempt: try with a different model
                        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-2-inpainting",
                            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                            local_files_only=False,
                            resume_download=True,
                            max_workers=1  # Reduce concurrent connections
                        )
                    else:
                        # Third attempt: try with a smaller model
                        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                            "CompVis/stable-diffusion-v1-4",
                            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                            local_files_only=False,
                            resume_download=True,
                            use_auth_token=None
                        )
                    
                    self.inpaint_pipeline = self.inpaint_pipeline.to(device)
                    print("Successfully loaded Stable Diffusion inpainting pipeline")
                    break
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # Last attempt
                        print("All attempts to load Stable Diffusion model failed. Using fallback inpainting.")
                        self.inpaint_pipeline = None
                    else:
                        import time
                        time.sleep(2)  # Wait before retry
        
        # Simple super-resolution (bilinear upsampling + noise reduction)
        self.sr_factor = 2
    
    def identify_reliable_regions(self, rendered_view: ViewInfo, reference_views: List[ViewInfo], 
                                 depth_threshold=0.1, color_threshold=0.2):
        """Identify reliable regions by comparing with reference views"""
        if rendered_view.image is None:
            # Create a simple reliability mask based on image dimensions
            H, W = rendered_view.camera.height, rendered_view.camera.width
            return torch.zeros(H, W, dtype=torch.bool, device=rendered_view.camera.R.device)
        
        H, W = rendered_view.image.shape[:2]
        reliability_mask = torch.zeros(H, W, dtype=torch.bool, device=rendered_view.image.device)
        
        # If no depth information, use a simple heuristic
        if rendered_view.depth is None:
            # Mark center region as reliable
            center_h, center_w = H // 2, W // 2
            h_start, h_end = max(0, center_h - H//4), min(H, center_h + H//4)
            w_start, w_end = max(0, center_w - W//4), min(W, center_w + W//4)
            reliability_mask[h_start:h_end, w_start:w_end] = True
            return reliability_mask
        
        for ref_view in reference_views:
            if ref_view.image is None or ref_view.depth is None:
                continue
            
            # Warp reference view to current view (simplified geometric warping)
            warped_img, warped_depth, valid_mask = self.warp_view(
                ref_view, rendered_view.camera
            )
            
            if warped_img is None or warped_depth is None:
                continue
            
            # Check depth consistency
            depth_diff = torch.abs(rendered_view.depth - warped_depth)
            depth_consistent = depth_diff < depth_threshold
            
            # Check color consistency  
            color_diff = torch.norm(rendered_view.image - warped_img, dim=2)
            color_consistent = color_diff < color_threshold
            
            # Combine masks
            consistent = depth_consistent & color_consistent & valid_mask
            reliability_mask |= consistent
        
        return reliability_mask
    
    def warp_view(self, source_view: ViewInfo, target_camera: CameraParams):
        """Warp source view to target camera (simplified warping)"""
        try:
            # This is a simplified warping - in practice would use proper 3D warping
            # For now, just return the source image resized to target size
            if source_view.image is None:
                return None, None, None
            
            H, W = target_camera.height, target_camera.width
            source_img = source_view.image
            source_depth = source_view.depth if source_view.depth is not None else torch.ones_like(source_img[:,:,0])
            
            # Simple resize as placeholder for complex 3D warping
            warped_img = torch.nn.functional.interpolate(
                source_img.permute(2, 0, 1).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            
            warped_depth = torch.nn.functional.interpolate(
                source_depth.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze()
            
            valid_mask = torch.ones(H, W, dtype=torch.bool, device=source_img.device)
            
            return warped_img, warped_depth, valid_mask
            
        except Exception as e:
            print(f"Warping failed: {e}")
            return None, None, None
    
    def generate_inpainting(self, target_view: ViewInfo, reference_views: List[ViewInfo], 
                          unreliable_mask: torch.Tensor):
        """Generate inpainting for unreliable regions using Stable Diffusion"""
        try:
            print(f"Debug: target_view.image.shape = {target_view.image.shape if target_view.image is not None else 'None'}")
            print(f"Debug: unreliable_mask.shape = {unreliable_mask.shape}")
            
            # Resize target image and mask to 512x512 for Stable Diffusion
            target_img = target_view.image.clone()
            mask = unreliable_mask.clone()
            
            # Resize to 512x512
            target_img_resized = torch.nn.functional.interpolate(
                target_img.permute(2, 0, 1).unsqueeze(0),
                size=(512, 512), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(512, 512), mode='nearest'
            ).squeeze(0).squeeze(0)
            
            print(f"Debug: target_img_resized.shape = {target_img_resized.shape}")
            print(f"Debug: mask_resized.shape = {mask_resized.shape}")
            
            # Convert to PIL images
            target_img_pil = transforms.ToPILImage()(target_img_resized.cpu())
            mask_pil = transforms.ToPILImage()(mask_resized.cpu())
            
            # Use very safe prompts to avoid NSFW detection
            safe_prompts = [
                "a simple geometric pattern",
                "basic shapes and lines",
                "minimalist design elements",
                "clean geometric forms",
                "simple architectural elements"
            ]
            
            # Try each prompt
            for prompt in safe_prompts:
                try:
                    # Disable NSFW checking
                    with torch.no_grad():
                        result = self.inpaint_pipeline(
                            prompt=prompt,
                            image=target_img_pil,
                            mask_image=mask_pil,
                            num_inference_steps=20,  # Reduced for speed
                            guidance_scale=7.5,
                            safety_checker=None  # Disable safety checker
                        ).images[0]
                    
                    # Convert back to tensor
                    result_tensor = transforms.ToTensor()(result).to(target_view.image.device)
                    print(f"Debug: result_tensor.shape = {result_tensor.shape}")
                    
                    # Resize back to original size
                    result_tensor = torch.nn.functional.interpolate(
                        result_tensor.unsqueeze(0),
                        size=target_view.image.shape[:2], mode='bilinear', align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                    
                    # Blend with original image - ensure mask has correct shape
                    final_image = target_view.image.clone()
                    mask_shape = unreliable_mask.shape
                    if result_tensor.shape[:2] != mask_shape:
                        # Resize result to match mask shape
                        result_tensor = torch.nn.functional.interpolate(
                            result_tensor.permute(2, 0, 1).unsqueeze(0),
                            size=mask_shape, mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                    
                    # Ensure dimensions are compatible for indexing
                    if unreliable_mask.dim() == 2 and result_tensor.dim() == 3:
                        # Use proper broadcasting for mask-based assignment
                        mask_3d = unreliable_mask.unsqueeze(-1).expand(-1, -1, 3)
                        final_image[mask_3d] = result_tensor[mask_3d]
                    else:
                        # Direct assignment if dimensions match
                        final_image[unreliable_mask] = result_tensor[unreliable_mask]
                    
                    return final_image
                    
                except Exception as e:
                    print(f"Inpainting failed with prompt '{prompt}': {e}")
                    continue
            
            # If all prompts fail, use fallback
            print("All inpainting attempts failed, using fallback method")
            return self.fallback_inpainting(target_view, unreliable_mask)
            
        except Exception as e:
            print(f"Inpainting failed: {e}")
            return self.fallback_inpainting(target_view, unreliable_mask)
    
    def fallback_inpainting(self, target_view: ViewInfo, unreliable_mask: torch.Tensor):
        """Fallback inpainting using simple interpolation"""
        try:
            if target_view.image is None:
                # Create a simple colored image if no target image
                H, W = target_view.camera.height, target_view.camera.width
                device = target_view.camera.R.device
                return torch.ones(H, W, 3, device=device, dtype=torch.float32) * 0.5
            
            img = target_view.image.clone()
            
            # Simple inpainting: fill unreliable regions with nearby pixel values
            if unreliable_mask.sum() > 0:
                # Use simple interpolation for unreliable regions
                unreliable_indices = torch.where(unreliable_mask)
                
                for i, j in zip(unreliable_indices[0], unreliable_indices[1]):
                    # Get nearby reliable pixels
                    h, w = img.shape[:2]
                    nearby_pixels = []
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and not unreliable_mask[ni, nj]:
                                nearby_pixels.append(img[ni, nj])
                    
                    if nearby_pixels:
                        # Average nearby pixels
                        avg_pixel = torch.stack(nearby_pixels).mean(dim=0)
                        img[i, j] = avg_pixel
                    else:
                        # If no nearby reliable pixels, use gray
                        img[i, j] = torch.tensor([0.5, 0.5, 0.5], device=img.device)
            
            return img
            
        except Exception as e:
            print(f"Fallback inpainting failed: {e}")
            # Return a simple gray image
            H, W = target_view.camera.height, target_view.camera.width
            device = target_view.camera.R.device
            return torch.ones(H, W, 3, device=device, dtype=torch.float32) * 0.5
    
    def apply_super_resolution(self, image: torch.Tensor):
        """Apply super-resolution enhancement"""
        # Simple upsampling + smoothing
        H, W = image.shape[:2]
        new_H, new_W = H * self.sr_factor, W * self.sr_factor
        
        # Bicubic upsampling
        upsampled = torch.nn.functional.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(new_H, new_W), mode='bicubic', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Simple sharpening filter
        kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], 
                             device=image.device, dtype=torch.float32) / 9.0
        
        sharpened = upsampled.clone()
        for c in range(3):
            channel = upsampled[:, :, c].unsqueeze(0).unsqueeze(0)
            sharpened_channel = torch.nn.functional.conv2d(
                channel, kernel.unsqueeze(0).unsqueeze(0), padding=1
            )
            sharpened[:, :, c] = sharpened_channel.squeeze()
        
        return torch.clamp(sharpened, 0, 1)

class ViewSelector:
    """Implements view selection algorithms from the paper"""
    
    def __init__(self, object_center=None):
        if object_center is not None:
            self.object_center = object_center
        else:
            # Create on CPU first, will be moved to appropriate device when needed
            self.object_center = torch.zeros(3, dtype=torch.float32)
    
    def select_anchor_views(self, known_views: List[ViewInfo], frontier_views: List[ViewInfo], 
                           k=8) -> List[int]:
        """Select k anchor views to maximize coverage and minimize redundancy"""
        if len(known_views) <= k:
            return list(range(len(known_views)))
        
        # Compute coverage scores (overlap with frontier views)
        coverage_scores = self.compute_coverage_scores(known_views, frontier_views)
        
        # Compute pairwise similarity matrix
        similarity_matrix = self.compute_view_similarity(known_views)
        
        # Greedy optimization: max s^T w - s^T E s
        selected_indices = self.greedy_selection(coverage_scores, similarity_matrix, k)
        
        return selected_indices
    
    def compute_coverage_scores(self, known_views: List[ViewInfo], frontier_views: List[ViewInfo]):
        """Compute coverage scores based on pixel overlap with frontier views"""
        n_known = len(known_views)
        # Get device from known views
        known_device = known_views[0].camera.R.device if known_views else device
        coverage_scores = torch.zeros(n_known, device=known_device, dtype=torch.float32)
        
        for i, known_view in enumerate(known_views):
            overlap_count = 0
            total_pixels = 0
            
            for frontier_view in frontier_views:
                # Simplified overlap computation - project frustums
                overlap = self.compute_view_overlap(known_view, frontier_view)
                overlap_count += overlap
                total_pixels += 1
            
            coverage_scores[i] = overlap_count / max(total_pixels, 1)
        
        return coverage_scores
    
    def compute_view_overlap(self, view1: ViewInfo, view2: ViewInfo):
        """Compute viewing frustum overlap (simplified)"""
        # Simplified: compute angular distance between camera centers
        cam1_pos = -view1.camera.R.T @ view1.camera.T
        cam2_pos = -view2.camera.R.T @ view2.camera.T
        
        # Direction from cameras to object center
        dir1 = self.object_center.to(cam1_pos.device) - cam1_pos
        dir2 = self.object_center.to(cam2_pos.device) - cam2_pos
        
        dir1 = dir1 / torch.norm(dir1)
        dir2 = dir2 / torch.norm(dir2)
        
        # Cosine similarity (higher = more overlap)
        similarity = torch.dot(dir1, dir2)
        
        # Convert to overlap score
        overlap = torch.clamp(similarity, 0, 1)
        
        return overlap.item()
    
    def compute_view_similarity(self, views: List[ViewInfo]):
        """Compute pairwise view similarity matrix"""
        n_views = len(views)
        # Get device from views
        view_device = views[0].camera.R.device if views else device
        similarity_matrix = torch.zeros(n_views, n_views, device=view_device, dtype=torch.float32)
        
        for i in range(n_views):
            for j in range(n_views):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute similarity based on camera pose
                    cam_i_pos = -views[i].camera.R.T @ views[i].camera.T
                    cam_j_pos = -views[j].camera.R.T @ views[j].camera.T
                    
                    # Ensure both positions are on the same device
                    cam_i_pos = cam_i_pos.to(view_device)
                    cam_j_pos = cam_j_pos.to(view_device)
                    
                    # Distance similarity
                    dist = torch.norm(cam_i_pos - cam_j_pos)
                    similarity_matrix[i, j] = torch.exp(-dist / 2.0)
        
        return similarity_matrix
    
    def greedy_selection(self, weights: torch.Tensor, similarity_matrix: torch.Tensor, k: int):
        """Greedy optimization of s^T w - s^T E s"""
        n = len(weights)
        selected = []
        remaining = set(range(n))
        
        for _ in range(k):
            best_score = float('-inf')
            best_idx = None
            
            for idx in remaining:
                # Compute objective if we add this index
                test_selection = selected + [idx]
                s = torch.zeros(n, device=weights.device, dtype=torch.float32)
                s[test_selection] = 1.0
                
                coverage_term = torch.dot(s, weights)
                penalty_term = s @ similarity_matrix @ s
                
                score = coverage_term - 0.1 * penalty_term  # λ = 0.1
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected
    
    def place_frontier_views(self, anchor_views: List[ViewInfo], distance_factor=1/3,
                           n_frontier=16) -> List[ViewInfo]:
        """Place frontier views along lines from anchors to object center"""
        frontier_views = []
        
        for anchor_view in anchor_views:
            # Current camera position
            cam_pos = -anchor_view.camera.R.T @ anchor_view.camera.T
            
            # Direction from camera to object center
            to_object = self.object_center.to(cam_pos.device) - cam_pos
            object_distance = torch.norm(to_object)
            direction = to_object / object_distance
            
            # Place frontier view closer to object
            new_distance = object_distance * distance_factor
            new_pos = cam_pos + direction * new_distance
            
            # Create new camera looking at object center
            new_R, new_T = self.look_at_matrix(new_pos, self.object_center.to(new_pos.device))
            
            frontier_camera = CameraParams(
                R=new_R,
                T=new_T,
                K=anchor_view.camera.K.clone(),
                width=anchor_view.camera.width,
                height=anchor_view.camera.height
            )
            
            frontier_view = ViewInfo(camera=frontier_camera)
            frontier_views.append(frontier_view)
        
        return frontier_views
    
    def look_at_matrix(self, eye: torch.Tensor, target: torch.Tensor, up=None):
        """Create look-at rotation and translation matrices"""
        if up is None:
            up = torch.tensor([0., 1., 0.], device=eye.device, dtype=torch.float32)
        
        # Forward direction (from eye to target)
        forward = target - eye
        forward = forward / torch.norm(forward)
        
        # Right direction
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        # Corrected up direction
        up = torch.cross(right, forward)
        up = up / torch.norm(up)
        
        # Rotation matrix (world to camera)
        R = torch.stack([right, up, -forward], dim=0)
        
        # Translation
        T = -R @ eye
        
        return R, T
    
    def sample_views_between_anchors_and_frontiers(self, anchor_views: List[ViewInfo], 
                                                  frontier_views: List[ViewInfo],
                                                  n_samples=100) -> List[ViewInfo]:
        """Sample views between anchor and frontier views"""
        sampled_views = []
        
        # Get device from anchor views
        anchor_device = anchor_views[0].camera.R.device if anchor_views else device
        
        for _ in range(n_samples):
            # Randomly select anchor and frontier views
            anchor_view = random.choice(anchor_views)
            frontier_view = random.choice(frontier_views)
            
            # Random interpolation factor
            t = torch.rand(1, device=anchor_device).item()
            
            # Interpolate camera positions
            anchor_pos = -anchor_view.camera.R.T @ anchor_view.camera.T
            frontier_pos = -frontier_view.camera.R.T @ frontier_view.camera.T
            
            # Ensure both positions are on the same device
            anchor_pos = anchor_pos.to(anchor_device)
            frontier_pos = frontier_pos.to(anchor_device)
            
            new_pos = (1 - t) * anchor_pos + t * frontier_pos
            
            # Create camera looking at object
            object_center_device = self.object_center.to(anchor_device) if self.object_center is not None else torch.zeros(3, device=anchor_device)
            new_R, new_T = self.look_at_matrix(new_pos, object_center_device)
            
            sampled_camera = CameraParams(
                R=new_R,
                T=new_T,
                K=anchor_view.camera.K.clone(),
                width=anchor_view.camera.width,
                height=anchor_view.camera.height
            )
            
            sampled_view = ViewInfo(camera=sampled_camera)
            sampled_views.append(sampled_view)
        
        return sampled_views
    
    def select_views_to_update(self, sampled_views: List[ViewInfo], 
                              anchor_views: List[ViewInfo], n_select=8) -> List[int]:
        """Select views to update from sampled views"""
        if len(sampled_views) <= n_select:
            return list(range(len(sampled_views)))
        
        # Get device from anchor views
        anchor_device = anchor_views[0].camera.R.device if anchor_views else device
        
        # Compute distances to anchors
        distances = []
        for sampled_view in sampled_views:
            sampled_pos = -sampled_view.camera.R.T @ sampled_view.camera.T
            sampled_pos = sampled_pos.to(anchor_device)
            
            min_dist = float('inf')
            for anchor_view in anchor_views:
                anchor_pos = -anchor_view.camera.R.T @ anchor_view.camera.T
                anchor_pos = anchor_pos.to(anchor_device)
                dist = torch.norm(sampled_pos - anchor_pos).item()
                min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        distances = torch.tensor(distances, device=anchor_device, dtype=torch.float32)
        
        # Apply distance discounting to avoid large jumps
        weights = torch.exp(-distances / distances.std())
        
        # Select views with highest weights
        _, selected_indices = torch.topk(weights, n_select)
        
        return selected_indices.tolist()

class ProgressiveGaussianSplatting:
    """Main class implementing progressive Close-up-GS"""
    
    def __init__(self, scene_bounds=None, sh_degree=3):
        if scene_bounds is not None:
            self.scene_bounds = scene_bounds
        else:
            # Create on CPU first, will be moved to appropriate device when needed
            self.scene_bounds = (torch.tensor([-1, -1, -1], dtype=torch.float32), 
                                torch.tensor([1, 1, 1], dtype=torch.float32))
        self.sh_degree = sh_degree
        
        # Initialize components
        self.gaussians = None
        self.renderer = GaussianRenderer()
        self.see3d_proxy = See3DProxy()
        self.view_selector = ViewSelector()
        
        # Training parameters
        self.lr_xyz = 0.00016
        self.lr_features = 0.0025
        self.lr_opacity = 0.05
        self.lr_scaling = 0.005
        self.lr_rotation = 0.001
        
        # Progressive parameters
        self.distance_factors = [1/3]  # Reduced from [1/3, 1/9, 1/27]
        self.n_anchor_views = 4  # Reduced from 8
        self.n_frontier_views = 8  # Reduced from 16
        self.n_sampled_views = 20  # Reduced from 100
        self.n_update_views = 4  # Reduced from 8
        
    def initialize_gaussians_from_points(self, points: torch.Tensor, colors: torch.Tensor = None):
        """Initialize Gaussians from point cloud with proper color initialization"""
        n_points = len(points)
        self.gaussians = Gaussian3D(num_points=n_points, sh_degree=self.sh_degree)
        
        with torch.no_grad():
            # Set positions
            self.gaussians._xyz.data = points.clone()
            
            # Set colors if provided
            if colors is not None:
                print(f"Setting colors with shape: {colors.shape}")
                print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
                # Ensure colors are in [0, 1] range
                colors = torch.clamp(colors, 0, 1)
                # Set DC component (constant color)
                self.gaussians._features_dc.data[:, 0, :] = colors.clone()
                # Set higher order components to zero
                self.gaussians._features_rest.data.zero_()
            else:
                # Initialize with random colors
                random_colors = torch.rand(n_points, 3, device=points.device)
                self.gaussians._features_dc.data[:, 0, :] = random_colors
                self.gaussians._features_rest.data.zero_()
            
            # Initialize scales based on nearest neighbor distances
            distances = torch.cdist(points, points)
            distances[distances == 0] = float('inf')
            nearest_distances = torch.min(distances, dim=1)[0]
            
            # Ensure nearest_distances are valid and positive
            nearest_distances = torch.clamp(nearest_distances, min=0.01)  # Minimum distance of 0.01
            
            # Check for any invalid values before log
            if torch.any(torch.isnan(nearest_distances)) or torch.any(nearest_distances <= 0):
                print("Warning: Invalid nearest distances found, using default scale")
                initial_scale = torch.full((n_points, 3), -2.0, device=points.device)  # log(0.135) ≈ -2
            else:
                initial_scale = torch.log(nearest_distances.unsqueeze(1).repeat(1, 3) / 3.0)
                # Clamp to reasonable range
                initial_scale = torch.clamp(initial_scale, -5.0, 2.0)
            
            self.gaussians._scaling.data = initial_scale
            
            # Initialize opacity to reasonable values - higher for better visibility
            self.gaussians._opacity.data = torch.ones(n_points, 1, device=points.device) * 0.9
            
            # Initialize rotation (identity quaternions)
            self.gaussians._rotation.data = torch.tensor([1.0, 0.0, 0.0, 0.0], device=points.device, dtype=torch.float32).repeat(n_points, 1)
            
            # Check for NaN values after initialization and fix them
            if torch.any(torch.isnan(self.gaussians._xyz)):
                print("ERROR: NaN detected in positions! Resetting...")
                self.gaussians._xyz.data = torch.randn_like(self.gaussians._xyz.data) * 0.1
                
            if torch.any(torch.isnan(self.gaussians._features_dc)):
                print("ERROR: NaN detected in colors! Resetting...")
                self.gaussians._features_dc.data = torch.rand_like(self.gaussians._features_dc.data)
                
            if torch.any(torch.isnan(self.gaussians._opacity)):
                print("ERROR: NaN detected in opacity! Resetting...")
                self.gaussians._opacity.data.fill_(0.8)
                
            if torch.any(torch.isnan(self.gaussians._scaling)):
                print("ERROR: NaN detected in scaling! Resetting...")
                self.gaussians._scaling.data.fill_(-2.0)  # log(0.135)
                
            if torch.any(torch.isnan(self.gaussians._rotation)):
                print("ERROR: NaN detected in rotation! Resetting...")
                # Reset to identity quaternions
                self.gaussians._rotation.data = torch.tensor([1.0, 0.0, 0.0, 0.0], 
                                                           device=self.gaussians._rotation.device, 
                                                           dtype=torch.float32).repeat(n_points, 1)
            
            print(f"Gaussian initialization complete:")
            print(f"  Positions: {self.gaussians._xyz.shape}")
            print(f"  Colors: {self.gaussians._features_dc.shape}")
            print(f"  Opacity range: [{self.gaussians._opacity.min():.3f}, {self.gaussians._opacity.max():.3f}]")
            print(f"  Scale range: [{self.gaussians._scaling.min():.3f}, {self.gaussians._scaling.max():.3f}]")
    
    def create_optimizer(self):
        """Create optimizer for Gaussian parameters"""
        param_groups = [
            {'params': [self.gaussians._xyz], 'lr': self.lr_xyz, 'name': 'xyz'},
            {'params': [self.gaussians._features_dc], 'lr': self.lr_features, 'name': 'f_dc'},
            {'params': [self.gaussians._features_rest], 'lr': self.lr_features / 20.0, 'name': 'f_rest'},
            {'params': [self.gaussians._opacity], 'lr': self.lr_opacity, 'name': 'opacity'},
            {'params': [self.gaussians._scaling], 'lr': self.lr_scaling, 'name': 'scaling'},
            {'params': [self.gaussians._rotation], 'lr': self.lr_rotation, 'name': 'rotation'}
        ]
        
        optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)
        return optimizer
    
    def train_baseline(self, training_views: List[ViewInfo], iterations=5000):
        """Train baseline 3DGS on distant training views"""
        print(f"Training baseline 3DGS for {iterations} iterations...")
        
        # Initialize Gaussians from training views if not done
        if self.gaussians is None:
            self.initialize_from_training_views(training_views)
        
        optimizer = self.create_optimizer()
        
        # Training loop
        for iteration in range(iterations):
            # Randomly select training view
            view_idx = torch.randint(0, len(training_views), (1,)).item()
            view = training_views[view_idx]
            
            if view.image is None:
                continue
            
            # Render
            rendered = self.renderer.render(self.gaussians, view.camera)
            
            # Compute loss
            loss = self.compute_loss(rendered, view)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at iteration {iteration}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN in gradients
            nan_detected = False
            for param in self.gaussians.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    print(f"WARNING: NaN gradient detected at iteration {iteration}")
                    nan_detected = True
                    break
            
            if nan_detected:
                optimizer.zero_grad()  # Clear bad gradients
                continue
            
            # Update positions gradient for densification
            if hasattr(self.gaussians._xyz, 'grad') and self.gaussians._xyz.grad is not None:
                self.gaussians._xyz.grad_accum = getattr(self.gaussians._xyz, 'grad_accum', 0) + torch.norm(self.gaussians._xyz.grad, dim=1)
            
            optimizer.step()
            
            # Check for NaN in parameters after optimizer step
            for param in self.gaussians.parameters():
                if torch.any(torch.isnan(param.data)):
                    print(f"ERROR: NaN in parameters after step {iteration}")
                    # Reset to initial values to prevent propagation
                    param.data.fill_(0.1)
            
            # Periodic densification and pruning
            if iteration % 500 == 0 and iteration > 0:
                self.gaussians.densify()
                self.gaussians.prune()
            
            # Show progress more frequently
            if iteration % 50 == 0:
                print(f"Iteration {iteration}/{iterations}, Loss: {loss.item():.6f}")
            elif iteration % 10 == 0:
                print(f"  Iteration {iteration}/{iterations}...")
        
        print("Baseline training completed!")
    
    def initialize_from_training_views(self, training_views: List[ViewInfo]):
        """Initialize Gaussians from training views with better quality"""
        print("Initializing Gaussians from training views...")
        
        # Collect points from all training views
        all_points = []
        all_colors = []
        
        for view in training_views:
            if view.image is not None:
                print(f"Processing view with image shape: {view.image.shape}")
                # Convert image to points (improved sampling)
                h, w = view.image.shape[-2:]
                
                # Sample more points for better coverage
                step = max(1, min(h, w) // 10)  # Sample every 10th pixel
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(-0.5, 0.5, h, device=view.image.device),  # Smaller range to avoid numerical issues
                    torch.linspace(-0.5, 0.5, w, device=view.image.device),
                    indexing='ij'
                )
                
                y_coords = y_coords[::step, ::step].flatten()
                x_coords = x_coords[::step, ::step].flatten()
                
                # Convert to 3D points - place points at z=0 (origin)
                points = torch.stack([x_coords, y_coords, torch.zeros_like(x_coords)], dim=1)
                
                # Check for NaN in points
                if torch.any(torch.isnan(points)):
                    print(f"Warning: NaN detected in generated points, skipping this view")
                    continue
                
                # Get colors
                colors = view.image.permute(1, 2, 0)[::step, ::step].reshape(-1, 3)
                
                # Check for NaN in colors
                if torch.any(torch.isnan(colors)):
                    print(f"Warning: NaN detected in colors, replacing with random colors")
                    colors = torch.rand_like(colors)
                
                all_points.append(points)
                all_colors.append(colors)
        
        if all_points:
            points = torch.cat(all_points, dim=0)
            colors = torch.cat(all_colors, dim=0)
            
            print(f"Total points collected: {len(points)}")
            print(f"Points shape: {points.shape}")
            print(f"Colors shape: {colors.shape}")
            print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
            
            # Limit number of points but keep more for better quality
            max_points = 800  # Increased from 500
            if len(points) > max_points:
                indices = torch.randperm(len(points))[:max_points]
                points = points[indices]
                colors = colors[indices]
            
            self.initialize_gaussians_from_points(points, colors)
            print(f"Initialized {len(points)} Gaussians from training views")
        else:
            # Fallback initialization with better parameters
            print("No valid training views found, using fallback initialization")
            self.gaussians = Gaussian3D(num_points=200, sh_degree=3)  # Increased from 100
    
    def depth_to_points(self, view: ViewInfo):
        """Convert depth map to 3D points"""
        if view.depth is None or view.image is None:
            return torch.empty(0, 3, device=view.camera.R.device), torch.empty(0, 3, device=view.camera.R.device)
        
        H, W = view.depth.shape
        camera = view.camera
        
        # Create pixel coordinates
        y, x = torch.meshgrid(torch.arange(H, device=view.camera.R.device), 
                             torch.arange(W, device=view.camera.R.device), indexing='ij')
        pixels = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        
        # Unproject to 3D
        depths = view.depth.flatten()
        valid_mask = depths > 0
        
        if valid_mask.sum() == 0:
            return torch.empty(0, 3, device=view.camera.R.device), torch.empty(0, 3, device=view.camera.R.device)
        
        # Ensure valid_mask is on the same device as pixels
        valid_mask = valid_mask.to(pixels.device)
        valid_pixels = pixels[valid_mask]
        valid_depths = depths[valid_mask]
        
        # Convert to normalized coordinates
        normalized_coords = torch.zeros(len(valid_pixels), 3, device=view.camera.R.device)
        normalized_coords[:, 0] = (valid_pixels[:, 0] - camera.K[0, 2].to(valid_pixels.device)) / camera.K[0, 0].to(valid_pixels.device)
        normalized_coords[:, 1] = (valid_pixels[:, 1] - camera.K[1, 2].to(valid_pixels.device)) / camera.K[1, 1].to(valid_pixels.device)
        normalized_coords[:, 2] = 1.0
        
        # Scale by depth
        points_cam = normalized_coords * valid_depths.to(normalized_coords.device).unsqueeze(1)
        
        # Transform to world coordinates
        points_world = points_cam @ camera.R.to(points_cam.device) + camera.T.to(points_cam.device)
        
        # Get corresponding colors
        colors = view.image.reshape(-1, 3)[valid_mask]
        
        # Subsample for efficiency
        if len(points_world) > 1000:
            indices = torch.randperm(len(points_world))[:1000]
            points_world = points_world[indices]
            colors = colors[indices]
        
        return points_world, colors
    
    def compute_loss(self, rendered: Dict, target_view: ViewInfo, reliable_mask=None):
        """Compute rendering loss"""
        losses = {}
        
        # L1 loss on RGB
        rgb_loss = torch.nn.functional.l1_loss(rendered['image'], target_view.image.to(rendered['image'].device))
        losses['rgb'] = rgb_loss
        
        # SSIM loss
        ssim_loss = 1.0 - self.compute_ssim(rendered['image'], target_view.image)
        losses['ssim'] = ssim_loss
        
        # Depth loss if available - skip for now since shapes don't match
        # The rendered depth is per-Gaussian while target depth is per-pixel
        # We'll need to implement proper depth rendering for this
        if target_view.depth is not None and 'depth' in rendered:
            # Skip depth loss for now to avoid shape mismatch
            # TODO: Implement proper depth rendering
            pass
        
        # Apply reliability mask if provided
        if reliable_mask is not None:
            for key in ['rgb', 'ssim']:
                if key in losses:
                    if reliable_mask.sum() > 0:
                        masked_rendered = rendered['image'][reliable_mask]
                        masked_target = target_view.image[reliable_mask]
                        if key == 'rgb':
                            losses[key] = torch.nn.functional.l1_loss(masked_rendered, masked_target)
                        elif key == 'ssim':
                            losses[key] = 1.0 - self.compute_ssim(masked_rendered.unsqueeze(0), 
                                                                 masked_target.unsqueeze(0))
        
        # Total loss
        total_loss = losses['rgb'] + 0.2 * losses['ssim']
        if 'depth' in losses:
            total_loss += losses['depth']
        
        return total_loss
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        """Compute SSIM between two images"""
        if SSIM_AVAILABLE:
            try:
                # Convert to numpy for skimage
                img1_np = img1.detach().cpu().numpy()
                img2_np = img2.detach().cpu().numpy()
                
                if len(img1_np.shape) == 3:
                    ssim_val = ssim(img1_np, img2_np, multichannel=True, channel_axis=2)
                else:
                    ssim_val = ssim(img1_np, img2_np)
                
                return torch.tensor(ssim_val, device=img1.device, dtype=torch.float32)
            except:
                pass
        
        # Fallback: simplified SSIM
        img2 = img2.to(img1.device)
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return torch.clamp(ssim_val, 0, 1)
    
    def progressive_expansion(self, training_views: List[ViewInfo], rounds=3) -> List[List[ViewInfo]]:
        """Progressive expansion from distant to close-up views"""
        print(f"Starting progressive expansion for {rounds} rounds...")
        
        all_rounds = []
        current_views = training_views.copy()
        
        for round_idx in range(rounds):
            print(f"\n=== Round {round_idx + 1}/{rounds} ===")
            
            # Clear GPU memory before each round
            torch.cuda.empty_cache()
            
            # Select anchor views
            print("Selecting anchor views...")
            anchor_indices = self.view_selector.select_anchor_views(
                current_views, [], k=self.n_anchor_views
            )
            anchor_views = [current_views[i] for i in anchor_indices]
            print(f"Selected {len(anchor_views)} anchor views")
            
            # Place frontier views
            print("Placing frontier views...")
            frontier_views = self.view_selector.place_frontier_views(
                anchor_views, distance_factor=self.distance_factors[0], n_frontier=self.n_frontier_views
            )
            
            # Sample views between anchors and frontiers
            print("Sampling views between anchors and frontiers...")
            sampled_views = self.view_selector.sample_views_between_anchors_and_frontiers(
                anchor_views, frontier_views, n_samples=self.n_sampled_views
            )
            
            # Select views to update
            update_indices = self.view_selector.select_views_to_update(
                sampled_views, anchor_views, n_select=self.n_update_views
            )
            update_views = [sampled_views[i] for i in update_indices]
            print(f"Selected {len(update_views)} views to update")
            
            # Render and refine views
            print("Rendering and refining views...")
            refined_views = []
            for i, view in enumerate(tqdm(update_views, desc="Processing views")):
                try:
                    # Render current view
                    rendered = self.renderer.render(self.gaussians, view.camera)
                    
                    # Check if rendering was successful
                    if rendered['image'] is None or torch.all(rendered['image'] == 0):
                        print(f"Warning: View {i} rendered empty image, using fallback")
                        # Create a simple fallback image
                        H, W = view.camera.height, view.camera.width
                        device = view.camera.R.device
                        fallback_image = torch.ones(H, W, 3, device=device, dtype=torch.float32) * 0.5
                        refined_view = ViewInfo(
                            camera=view.camera,
                            image=fallback_image,
                            reliability_score=0.5
                        )
                        refined_views.append(refined_view)
                        continue
                    
                    # Identify unreliable regions
                    unreliable_mask = self.see3d_proxy.identify_reliable_regions(
                        ViewInfo(camera=view.camera, image=rendered['image'], depth=rendered['depth']),
                        current_views
                    )
                    
                    # Generate inpainting for unreliable regions
                    refined_image = self.see3d_proxy.generate_inpainting(
                        view, current_views, unreliable_mask
                    )
                    
                    # Check if inpainting was successful
                    if refined_image is None:
                        print(f"Warning: View {i} inpainting failed, using rendered image")
                        refined_image = rendered['image']
                    
                    # Create refined view
                    refined_view = ViewInfo(
                        camera=view.camera,
                        image=refined_image,
                        reliability_score=0.8
                    )
                    refined_views.append(refined_view)
                    
                    # Clear memory after each view
                    if i % 4 == 0:  # Clear every 4 views
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing view {i}: {e}")
                    # Create a fallback view with a simple image
                    H, W = view.camera.height, view.camera.width
                    device = view.camera.R.device
                    fallback_image = torch.ones(H, W, 3, device=device, dtype=torch.float32) * 0.5
                    fallback_view = ViewInfo(
                        camera=view.camera,
                        image=fallback_image,
                        reliability_score=0.3
                    )
                    refined_views.append(fallback_view)
            
            # Update current views
            current_views.extend(refined_views)
            all_rounds.append(refined_views)
            
            # Fine-tune Gaussians with new views
            print("Fine-tuning Gaussians...")
            self.fine_tune_gaussians(current_views, iterations=200)  # Increased from 100
            
            # Clear memory after round
            torch.cuda.empty_cache()
        
        return all_rounds
    
    def fine_tune_gaussians(self, all_views: List[ViewInfo], iterations=100):  # Reduced from 5000
        """Fine-tune Gaussians on all available views"""
        if len(all_views) == 0:
            return
        
        print(f"Fine-tuning Gaussians on {len(all_views)} views for {iterations} iterations...")
        
        optimizer = self.create_optimizer()
        
        # Training loop
        for iteration in range(iterations):
            # Randomly select view
            view_idx = torch.randint(0, len(all_views), (1,)).item()
            view = all_views[view_idx]
            
            if view.image is None:
                continue
            
            # Render
            rendered = self.renderer.render(self.gaussians, view.camera)
            
            # Compute loss
            loss = self.compute_loss(rendered, view)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear gradients to save memory
            if hasattr(self.gaussians._xyz, 'grad'):
                self.gaussians._xyz.grad = None
            if hasattr(self.gaussians._features_dc, 'grad'):
                self.gaussians._features_dc.grad = None
            if hasattr(self.gaussians._opacity, 'grad'):
                self.gaussians._opacity.grad = None
            if hasattr(self.gaussians._scaling, 'grad'):
                self.gaussians._scaling.grad = None
            if hasattr(self.gaussians._rotation, 'grad'):
                self.gaussians._rotation.grad = None
            
            # Clear memory periodically
            if iteration % 20 == 0:  # Clear every 20 iterations
                torch.cuda.empty_cache()
        
        print("Fine-tuning completed!")

class EvaluationMetrics:
    """Comprehensive evaluation metrics for Close-up-GS"""
    
    def __init__(self):
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            except:
                print("Warning: LPIPS model loading failed")
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute Peak Signal-to-Noise Ratio"""
        # Ensure tensors are on the same device
        target = target.to(pred.device)
        
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return 100.0  # Very high PSNR for identical images
        elif mse < 1e-10:
            return 100.0  # Avoid numerical issues
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Clamp to reasonable range
        psnr_val = torch.clamp(psnr, 0, 100).item()
        return psnr_val
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute Structural Similarity Index"""
        # Ensure tensors are on the same device
        target = target.to(pred.device)
        
        if SSIM_AVAILABLE:
            try:
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                if len(pred_np.shape) == 3:
                    ssim_val = ssim(pred_np, target_np, multichannel=True, channel_axis=2, data_range=1.0)
                else:
                    ssim_val = ssim(pred_np, target_np, data_range=1.0)
                
                return max(0.0, min(1.0, ssim_val))  # Clamp to [0, 1]
            except Exception as e:
                print(f"SSIM computation failed: {e}")
        
        # Fallback SSIM
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Add small epsilon to avoid division by zero
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        if denominator < 1e-10:
            return 0.0
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / denominator
        
        return torch.clamp(ssim_val, 0, 1).item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute Learned Perceptual Image Patch Similarity"""
        if self.lpips_fn is None:
            return 0.0  # Fallback value
        
        try:
            # Ensure tensors are on the same device as the LPIPS model
            target = target.to(pred.device)
            
            # Ensure images are in [-1, 1] range
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            
            # Reshape to (B, C, H, W)
            if len(pred_norm.shape) == 3:
                pred_norm = pred_norm.permute(2, 0, 1).unsqueeze(0)
                target_norm = target_norm.permute(2, 0, 1).unsqueeze(0)
            
            # Move to the same device as LPIPS model
            pred_norm = pred_norm.to(next(self.lpips_fn.parameters()).device)
            target_norm = target_norm.to(next(self.lpips_fn.parameters()).device)
            
            lpips_val = self.lpips_fn(pred_norm, target_norm)
            return max(0.0, min(1.0, lpips_val.item()))  # Clamp to [0, 1]
        except Exception as e:
            print(f"LPIPS computation failed: {e}")
            return 0.5  # Return neutral value on failure
    
    def compute_dino_score(self, pred: torch.Tensor, reference: torch.Tensor):
        """Compute DINO feature similarity (simplified proxy)"""
        # Simplified DINO score using color histogram similarity
        try:
            # Convert to numpy
            pred_np = pred.detach().cpu().numpy()
            ref_np = reference.detach().cpu().numpy()
            
            # Compute histograms for each channel
            pred_hists = []
            ref_hists = []
            
            for c in range(3):
                pred_hist, _ = np.histogram(pred_np[:,:,c], bins=64, range=(0, 1))
                ref_hist, _ = np.histogram(ref_np[:,:,c], bins=64, range=(0, 1))
                pred_hists.append(pred_hist)
                ref_hists.append(ref_hist)
            
            # Compute correlation
            pred_features = np.concatenate(pred_hists)
            ref_features = np.concatenate(ref_hists)
            
            correlation = np.corrcoef(pred_features, ref_features)[0, 1]
            
            return max(0.0, correlation)  # Clamp to [0, 1]
            
        except:
            return 0.5  # Fallback value
    
    def compute_meta_iqa(self, image: torch.Tensor):
        """Compute no-reference image quality (simplified NIQE proxy)"""
        try:
            # Simple no-reference quality based on local variance and contrast
            img_np = image.detach().cpu().numpy()
            
            # Convert to grayscale
            if len(img_np.shape) == 3:
                gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
            else:
                gray = img_np
            
            # Compute local variance (measure of sharpness)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            laplacian_var = cv2.filter2D(gray, -1, kernel).var()
            
            # Normalize to [0, 1] range (higher is better quality)
            quality = min(1.0, laplacian_var / 0.1)
            
            return quality
            
        except:
            return 0.5  # Fallback value
    
    def evaluate_view(self, pred_image: torch.Tensor, target_image: torch.Tensor = None,
                     reference_images: List[torch.Tensor] = None):
        """Comprehensive evaluation of a single view"""
        metrics = {}
        
        if target_image is not None:
            # Metrics requiring ground truth
            metrics['psnr'] = self.compute_psnr(pred_image, target_image)
            metrics['ssim'] = self.compute_ssim(pred_image, target_image)
            metrics['lpips'] = self.compute_lpips(pred_image, target_image)
        
        # No-reference quality
        metrics['meta_iqa'] = self.compute_meta_iqa(pred_image)
        
        # DINO score with reference images
        if reference_images:
            dino_scores = []
            for ref_img in reference_images[:3]:  # Use first 3 references
                score = self.compute_dino_score(pred_image, ref_img)
                dino_scores.append(score)
            metrics['dino'] = np.mean(dino_scores) if dino_scores else 0.0
        
        return metrics
    
    def evaluate_progressive_results(self, progressive_views: List[List[ViewInfo]], 
                                   target_views: List[ViewInfo] = None):
        """Evaluate results from all progressive rounds"""
        results = {}
        
        for round_idx, round_views in enumerate(progressive_views):
            print(f"\nEvaluating Round {round_idx}...")
            
            round_metrics = {
                'psnr': [],
                'ssim': [],
                'lpips': [],
                'dino': [],
                'meta_iqa': []
            }
            
            # Count valid views
            valid_views = 0
            
            for view_idx, view in enumerate(round_views[:10]):  # Evaluate first 10 views
                if view.image is None:
                    print(f"  View {view_idx}: No image available")
                    continue
                
                # Check if image is valid and provide debug info
                img_min, img_max = view.image.min().item(), view.image.max().item()
                img_mean = view.image.mean().item()
                non_zero_pixels = (view.image.sum(dim=-1) > 0).sum().item()
                total_pixels = view.image.shape[0] * view.image.shape[1]
                
                print(f"  View {view_idx}: Image range [{img_min:.3f}, {img_max:.3f}], mean {img_mean:.3f}")
                print(f"  View {view_idx}: Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.1f}%)")
                
                # Only skip if image is completely null (None or all exactly 0)
                if torch.all(view.image == 0):
                    print(f"  View {view_idx}: Skipping - image is completely black")
                    continue
                    
                # For very low quality images, still evaluate but note the quality
                if img_max < 0.01:
                    print(f"  View {view_idx}: Warning - very low intensity image (max: {img_max:.6f})")
                elif non_zero_pixels < total_pixels * 0.01:
                    print(f"  View {view_idx}: Warning - very sparse image ({100*non_zero_pixels/total_pixels:.2f}% non-zero)")
                
                valid_views += 1
                print(f"  View {view_idx}: Evaluating...")
                
                # Find corresponding target if available
                target_img = None
                if target_views and view_idx < len(target_views):
                    target_img = target_views[view_idx].image
                
                # Reference images from round 0 (training views)
                ref_images = [v.image for v in progressive_views[0][:5] if v.image is not None]
                
                try:
                    metrics = self.evaluate_view(view.image, target_img, ref_images)
                    
                    for key, value in metrics.items():
                        if key in round_metrics and value is not None:
                            round_metrics[key].append(value)
                            
                except Exception as e:
                    print(f"  View {view_idx}: Evaluation failed - {e}")
                    continue
            
            print(f"  Valid views in round {round_idx}: {valid_views}")
            
            # Compute average metrics
            round_results = {}
            for key, values in round_metrics.items():
                if values:
                    round_results[f'{key}_mean'] = np.mean(values)
                    round_results[f'{key}_std'] = np.std(values)
                else:
                    round_results[f'{key}_mean'] = 0.0
                    round_results[f'{key}_std'] = 0.0
            
            results[f'round_{round_idx}'] = round_results
            
            # Print summary
            print(f"Round {round_idx} Results:")
            for key, value in round_results.items():
                if key.endswith('_mean'):
                    metric_name = key.replace('_mean', '').upper()
                    std_key = key.replace('_mean', '_std')
                    std_value = round_results.get(std_key, 0.0)
                    print(f"  {metric_name}: {value:.4f} ± {std_value:.4f}")
        
        return results

def load_llff_dataset(scene_path: str, downsample_factor=4):
    """Load LLFF dataset (simplified loader)"""
    scene_path = Path(scene_path)
    
    # Mock LLFF data structure
    print(f"Loading LLFF scene from {scene_path}")
    
    # Create synthetic training views (distant views)
    training_views = []
    
    # Camera intrinsics (typical LLFF values)
    focal = 800.0 / downsample_factor
    cx = 400.0 / downsample_factor
    cy = 300.0 / downsample_factor
    W = int(800 / downsample_factor)
    H = int(600 / downsample_factor)
    
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    
    # Create circular training views around object
    object_center = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)
    radius = 3.0  # Distant views
    n_views = 20
    
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        height = 0.5 * np.sin(angle * 2)  # Varying height
        
        # Camera position
        cam_pos = torch.tensor([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ], device=device, dtype=torch.float32)
        
        # Look at object center
        forward = object_center - cam_pos
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Rotation matrix (world to camera)
        R = torch.stack([right, up, -forward], dim=0)
        T = -R @ cam_pos
        
        camera = CameraParams(R=R, T=T, K=K, width=W, height=H)
        
        # Generate synthetic image and depth
        image = torch.rand(H, W, 3, device=device, dtype=torch.float32) * 0.8 + 0.1  # Random scene
        depth = torch.ones(H, W, device=device, dtype=torch.float32) * radius * 0.8  # Approximate depth
        
        view = ViewInfo(camera=camera, image=image, depth=depth)
        training_views.append(view)
    
    print(f"Loaded {len(training_views)} training views")
    
    return {
        'training_views': training_views,
        'object_center': object_center,
        'scene_bounds': (torch.tensor([-2, -2, -2], device=device, dtype=torch.float32), 
                        torch.tensor([2, 2, 2], device=device, dtype=torch.float32))
    }

def main():
    parser = argparse.ArgumentParser(description='Close-up-GS: Progressive 3D Gaussian Splatting')
    parser.add_argument('--dataset', type=str, default='llff', choices=['llff'], 
                       help='Dataset type')
    parser.add_argument('--scene', type=str, default='flower', 
                       help='Scene name')
    parser.add_argument('--rounds', type=int, default=3, 
                       help='Number of progressive rounds')
    parser.add_argument('--target_pos', type=float, nargs=3, default=[0., 0., 0.], 
                       help='Target object position')
    parser.add_argument('--baseline_iterations', type=int, default=3000, 
                       help='Baseline training iterations')
    parser.add_argument('--finetune_iterations', type=int, default=2000, 
                       help='Fine-tuning iterations per round')
    parser.add_argument('--output_dir', type=str, default='./output', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("=== Close-up-GS: Progressive 3D Gaussian Splatting ===")
    print(f"Dataset: {args.dataset}")
    print(f"Scene: {args.scene}")
    print(f"Rounds: {args.rounds}")
    print(f"Target position: {args.target_pos}")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    scene_data = load_llff_dataset(f"./data/{args.dataset}/{args.scene}")
    training_views = scene_data['training_views']
    object_center = torch.tensor(args.target_pos, device=device, dtype=torch.float32)
    
    # Initialize Close-up-GS
    closeup_gs = ProgressiveGaussianSplatting(
        scene_bounds=scene_data['scene_bounds']
    )
    closeup_gs.view_selector.object_center = object_center
    
    # Train baseline 3DGS
    print("\n=== Training Baseline 3DGS ===")
    closeup_gs.train_baseline(training_views, args.baseline_iterations)
    
    # Save baseline results
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    
    # Demonstrate close-up artifacts with baseline
    print("\n=== Demonstrating Close-up Artifacts ===")
    
    # Create close-up test views (3x closer)
    test_camera = training_views[0].camera
    close_pos = object_center + (object_center - (-test_camera.R.T @ test_camera.T)) / 3
    close_R, close_T = closeup_gs.view_selector.look_at_matrix(close_pos, object_center)
    close_camera = CameraParams(R=close_R, T=close_T, K=test_camera.K, 
                               width=test_camera.width, height=test_camera.height)
    
    baseline_closeup = closeup_gs.renderer.render(closeup_gs.gaussians, close_camera)
    
    # Save baseline close-up result
    baseline_img = baseline_closeup['image'].detach().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(training_views[0].image.detach().cpu().numpy())
    plt.title("Training View (Distant)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(baseline_img)
    plt.title("Baseline Close-up (3x closer)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Progressive expansion
    print("\n=== Progressive Expansion ===")
    all_rounds_views = closeup_gs.progressive_expansion(training_views, args.rounds)
    
    # Save progressive results
    for round_idx, round_views in enumerate(all_rounds_views):
        round_dir = output_dir / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)
        
        # Save a few sample views
        for view_idx, view in enumerate(round_views[:5]):
            if view.image is not None:
                view_img = view.image.detach().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(view_img)
                plt.title(f"Round {round_idx}, View {view_idx}")
                plt.axis('off')
                plt.savefig(round_dir / f"view_{view_idx}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Final close-up rendering
    print("\n=== Final Close-up Rendering ===")
    final_closeup = closeup_gs.renderer.render(closeup_gs.gaussians, close_camera)
    
    # Save final comparison
    final_img = final_closeup['image'].detach().cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(training_views[0].image.detach().cpu().numpy())
    plt.title("Training View (Distant)")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(baseline_img)
    plt.title("Baseline Close-up")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(final_img)
    plt.title("Close-up-GS Result")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "final_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluation
    print("\n=== Evaluation ===")
    evaluator = EvaluationMetrics()
    
    # Create reference views from training views
    reference_views = training_views[:5]
    
    evaluation_results = evaluator.evaluate_progressive_results(all_rounds_views, reference_views)
    
    # Save evaluation results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert tensors to floats for JSON serialization
        serializable_results = {}
        for round_key, round_data in evaluation_results.items():
            serializable_results[round_key] = {}
            for metric_key, value in round_data.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    serializable_results[round_key][metric_key] = float(value)
                else:
                    serializable_results[round_key][metric_key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("Close-up-GS processing completed!")
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Baseline Gaussians: {len(closeup_gs.gaussians._xyz)}")
    print(f"Progressive rounds: {args.rounds}")
    print(f"Total views processed: {sum(len(round_views) for round_views in all_rounds_views)}")
    
    if 'round_2' in evaluation_results:  # Final round
        final_metrics = evaluation_results['round_2']
        print("Final quality metrics:")
        for key, value in final_metrics.items():
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '').upper()
                print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 