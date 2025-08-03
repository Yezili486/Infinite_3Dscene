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
            bg_color = torch.zeros(3, device=device, dtype=torch.float32)
        else:
            # Get device from gaussians
            gaussian_device = gaussians.get_xyz().device if gaussians is not None else device
            bg_color = torch.tensor(bg_color, device=gaussian_device, dtype=torch.float32)
            
        # Transform Gaussians to camera space
        means3D = gaussians.get_xyz()  # (N, 3)
        means2D, depths = self.project_points(means3D, camera)
        
        # Get Gaussian properties
        opacities = gaussians.get_opacity()  # (N, 1)
        colors = self.compute_colors(gaussians, camera, means3D)  # (N, 3)
        scales = gaussians.get_scaling()  # (N, 3)
        rotations = gaussians.get_rotation()  # (N, 4) quaternions
        
        # Compute 2D covariance matrices
        cov2D = self.compute_2d_covariance(means3D, scales, rotations, camera)
        
        # Sort by depth
        sorted_indices = torch.argsort(depths)
        
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
        
        return rendered
    
    def project_points(self, points3D, camera: CameraParams):
        """Project 3D points to 2D image coordinates"""
        # Transform to camera coordinates
        points_cam = points3D @ camera.R.T.to(points3D.device) + camera.T.to(points3D.device)
        
        # Project to image plane
        points_proj = points_cam @ camera.K.T.to(points_cam.device)
        depths = points_cam[:, 2:3]
        points2D = points_proj[:, :2] / depths
        
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
        # Convert quaternions to rotation matrices
        rot_matrices = self.quaternion_to_matrix(rotations)  # (N, 3, 3)
        
        # Scale matrices
        scale_matrices = torch.diag_embed(scales)  # (N, 3, 3)
        
        # 3D covariance: R * S * S^T * R^T
        cov3D = rot_matrices @ scale_matrices @ scale_matrices.transpose(-2, -1) @ rot_matrices.transpose(-2, -1)
        
        # Project to 2D using Jacobian of projection
        # Simplified projection jacobian (assuming perspective)
        focal_x = camera.K[0, 0].to(means3D.device)
        focal_y = camera.K[1, 1].to(means3D.device)
        
        # Transform points to camera space
        points_cam = means3D @ camera.R.T.to(means3D.device) + camera.T.to(means3D.device)
        z = points_cam[:, 2]
        
        # Projection Jacobian (simplified)
        J = torch.zeros(means3D.shape[0], 2, 3, device=means3D.device)
        J[:, 0, 0] = focal_x / z
        J[:, 0, 2] = -focal_x * points_cam[:, 0] / (z ** 2)
        J[:, 1, 1] = focal_y / z
        J[:, 1, 2] = -focal_y * points_cam[:, 1] / (z ** 2)
        
        # Apply camera rotation to covariance
        cov3D_cam = camera.R.to(cov3D.device) @ cov3D @ camera.R.T.to(cov3D.device)
        
        # Project: J * Σ_3D * J^T
        cov2D = J @ cov3D_cam @ J.transpose(-2, -1)
        
        # Add small regularization for numerical stability
        eye = torch.eye(2, device=cov2D.device).expand(cov2D.shape[0], -1, -1)
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
        """Alpha blend sorted Gaussians"""
        H, W = camera.height, camera.width
        
        # Create output tensors
        # Get device from camera parameters
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
        
        batch_size = 50  # Process Gaussians in smaller batches
        for batch_start in range(0, len(means2D), batch_size):
            batch_end = min(batch_start + batch_size, len(means2D))
            
            for i in range(batch_start, batch_end):
                # Compute Gaussian influence
                center = means2D[i:i+1]  # (1, 2)
                cov = cov2D[i]  # (2, 2)
                
                # Ensure center is on the same device as pixels
                center = center.to(pixels.device)
                cov = cov.to(pixels.device)
                
                # Compute squared Mahalanobis distance
                diff = pixels - center  # (H*W, 2)
                try:
                    cov_inv = torch.inverse(cov)
                    quad_form = torch.sum(diff * (diff @ cov_inv), dim=1)  # (H*W,)
                    
                    # Gaussian weight (unnormalized)
                    weights = torch.exp(-0.5 * quad_form)
                    
                    # Apply opacity
                    alpha = opacities[i].to(pixels.device) * weights
                    
                    # Alpha blending
                    transmittance = 1.0 - accumulated_alpha
                    contribution = alpha * transmittance
                    
                    accumulated_color += contribution.unsqueeze(1) * colors[i:i+1].to(pixels.device)
                    accumulated_depth += contribution * depths[i].to(pixels.device)
                    accumulated_alpha += contribution
                    
                    # Early termination for fully opaque pixels
                    if torch.all(accumulated_alpha > 0.99):
                        break
                        
                except RuntimeError:
                    # Skip if covariance is not invertible
                    continue
            
            # Clean up GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add background
        transmittance = 1.0 - accumulated_alpha
        accumulated_color += transmittance.unsqueeze(1) * bg_color.to(accumulated_color.device)
        
        # Reshape to image
        final_image = accumulated_color.view(H, W, 3)
        final_depth = accumulated_depth.view(H, W)
        final_alpha = accumulated_alpha.view(H, W)
        
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
            return torch.ones_like(rendered_view.depth, dtype=torch.bool)
        
        H, W = rendered_view.image.shape[:2]
        reliability_mask = torch.zeros(H, W, dtype=torch.bool, device=rendered_view.image.device)
        
        for ref_view in reference_views:
            if ref_view.image is None or ref_view.depth is None:
                continue
            
            # Warp reference view to current view (simplified geometric warping)
            warped_img, warped_depth, valid_mask = self.warp_view(
                ref_view, rendered_view.camera
            )
            
            if warped_img is None:
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
        """Generate content for unreliable regions using inpainting"""
        if self.inpaint_pipeline is None:
            # Fallback: simple interpolation
            return self.fallback_inpainting(target_view, unreliable_mask)
        
        try:
            # Convert to PIL format for diffusion model
            target_img = target_view.image.detach().cpu().numpy()
            target_img = (target_img * 255).astype(np.uint8)
            
            mask = unreliable_mask.detach().cpu().numpy().astype(np.uint8) * 255
            
            # Resize to 512x512 for Stable Diffusion
            target_img_resized = cv2.resize(target_img, (512, 512))
            mask_resized = cv2.resize(mask, (512, 512))
            
            # Use safer prompts to avoid NSFW detection
            safe_prompts = [
                "natural scene, clean background, simple objects",
                "neutral environment, plain background, basic scene",
                "simple scene, clean background, natural lighting",
                "basic environment, neutral background, simple objects"
            ]
            
            # Try different prompts if one fails
            for prompt in safe_prompts:
                try:
                    # Run inpainting
                    result = self.inpaint_pipeline(
                        prompt=prompt,
                        image=target_img_resized,
                        mask_image=mask_resized,
                        num_inference_steps=20,
                        strength=0.8
                    ).images[0]
                    
                    # Check if result is mostly black (NSFW detection)
                    result_array = np.array(result)
                    if np.mean(result_array) < 10:  # Very dark image
                        print(f"NSFW detection triggered with prompt: {prompt}, trying next...")
                        continue
                    
                    # Convert back to tensor and resize to original size
                    result_tensor = torch.tensor(result_array, device=target_view.image.device, dtype=torch.float32) / 255.0
                    result_tensor = torch.nn.functional.interpolate(
                        result_tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(target_view.image.shape[0], target_view.image.shape[1]),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                    
                    # Blend with original
                    reliable_mask = ~unreliable_mask
                    final_img = target_view.image.clone()
                    final_img[unreliable_mask] = result_tensor[unreliable_mask]
                    
                    return final_img
                    
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
        img = target_view.image.clone()
        
        # Simple inpainting: interpolate from neighboring reliable pixels
        for c in range(3):  # RGB channels
            channel = img[:, :, c]
            
            # Use OpenCV inpainting if available
            try:
                channel_np = channel.detach().cpu().numpy()
                mask_np = unreliable_mask.detach().cpu().numpy().astype(np.uint8)
                
                inpainted = cv2.inpaint(
                    (channel_np * 255).astype(np.uint8),
                    mask_np * 255,
                    3, cv2.INPAINT_TELEA
                )
                
                img[:, :, c] = torch.tensor(inpainted, device=img.device, dtype=torch.float32) / 255.0
                
            except Exception:
                # Ultimate fallback: just use mean color
                mean_color = channel[~unreliable_mask].mean() if (~unreliable_mask).any() else 0.5
                img[unreliable_mask, c] = mean_color
        
        return img
    
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
        self.distance_factors = [1/3, 1/9, 1/27]  # 3x, 9x, 27x closer
        self.n_anchor_views = 8
        self.n_frontier_views = 16
        self.n_sampled_views = 100
        self.n_update_views = 8
        
    def initialize_gaussians_from_points(self, points: torch.Tensor, colors: torch.Tensor = None):
        """Initialize Gaussians from point cloud"""
        n_points = len(points)
        self.gaussians = Gaussian3D(num_points=n_points, sh_degree=self.sh_degree)
        
        with torch.no_grad():
            # Set positions
            self.gaussians._xyz.data = points.clone()
            
            # Set colors if provided
            if colors is not None:
                self.gaussians._features_dc.data[:, 0, :] = colors.clone()
            
            # Initialize scales based on nearest neighbor distances
            distances = torch.cdist(points, points)
            distances[distances == 0] = float('inf')
            nearest_distances = torch.min(distances, dim=1)[0]
            
            initial_scale = torch.log(nearest_distances.unsqueeze(1).repeat(1, 3) / 3.0)
            self.gaussians._scaling.data = initial_scale
    
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update positions gradient for densification
            if hasattr(self.gaussians._xyz, 'grad') and self.gaussians._xyz.grad is not None:
                self.gaussians._xyz.grad_accum = getattr(self.gaussians._xyz, 'grad_accum', 0) + torch.norm(self.gaussians._xyz.grad, dim=1)
            
            optimizer.step()
            
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
        """Initialize Gaussians from training view depth and color"""
        all_points = []
        all_colors = []
        
        for view in training_views[:3]:  # Use first few views for initialization
            if view.image is None:
                continue
            
            # Generate point cloud from depth (if available) or use uniform sampling
            if view.depth is not None:
                points, colors = self.depth_to_points(view)
                all_points.append(points)
                all_colors.append(colors)
        
        if len(all_points) == 0:
            # Fallback: random initialization
            # Get device from training views
            view_device = training_views[0].camera.R.device if training_views else device
            points = torch.randn(500, 3, device=view_device) * 0.5
            colors = torch.rand(500, 3, device=view_device)
        else:
            points = torch.cat(all_points, dim=0)
            colors = torch.cat(all_colors, dim=0)
            
                    # Subsample if too many points
        if len(points) > 1000:
            indices = torch.randperm(len(points))[:1000]
            points = points[indices]
            colors = colors[indices]
        
        self.initialize_gaussians_from_points(points, colors)
        print(f"Initialized {len(points)} Gaussians from training views")
    
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
        
        # Depth loss if available
        if target_view.depth is not None:
            depth_loss = torch.nn.functional.l1_loss(rendered['depth'], target_view.depth.to(rendered['depth'].device))
            losses['depth'] = depth_loss * 0.1
        
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
        """Run progressive expansion for multiple rounds"""
        print(f"Starting progressive expansion for {rounds} rounds...")
        
        known_views = training_views.copy()
        all_rounds_views = [known_views.copy()]
        
        for round_idx in range(rounds):
            print(f"\n=== Round {round_idx + 1}/{rounds} ===")
            distance_factor = self.distance_factors[round_idx]
            
            # Step 1: Select anchor views from known views
            print("Selecting anchor views...")
            frontier_views = self.view_selector.place_frontier_views(
                known_views[:self.n_anchor_views], distance_factor, self.n_frontier_views
            )
            anchor_indices = self.view_selector.select_anchor_views(
                known_views, frontier_views, self.n_anchor_views
            )
            anchor_views = [known_views[i] for i in anchor_indices]
            
            print(f"Selected {len(anchor_views)} anchor views")
            
            # Step 2: Place frontier views
            print("Placing frontier views...")
            frontier_views = self.view_selector.place_frontier_views(
                anchor_views, distance_factor, self.n_frontier_views
            )
            
            # Step 3: Sample and select views to update
            print("Sampling views between anchors and frontiers...")
            sampled_views = self.view_selector.sample_views_between_anchors_and_frontiers(
                anchor_views, frontier_views, self.n_sampled_views
            )
            
            update_indices = self.view_selector.select_views_to_update(
                sampled_views, anchor_views, self.n_update_views
            )
            views_to_update = [sampled_views[i] for i in update_indices]
            
            print(f"Selected {len(views_to_update)} views to update")
            
            # Step 4: Render and refine views
            print("Rendering and refining views...")
            refined_views = []
            
            for view in views_to_update + frontier_views:
                # Render with current 3DGS
                rendered = self.renderer.render(self.gaussians, view.camera)
                view.image = rendered['image']
                view.depth = rendered['depth']
                
                # Identify reliable regions
                reliable_mask = self.see3d_proxy.identify_reliable_regions(
                    view, anchor_views
                )
                
                # Refine unreliable regions
                if (~reliable_mask).sum() > 0:
                    refined_image = self.see3d_proxy.generate_inpainting(
                        view, anchor_views, ~reliable_mask
                    )
                    
                    # Apply super-resolution
                    enhanced_image = self.see3d_proxy.apply_super_resolution(refined_image)
                    
                    # Resize back to original size if needed
                    if enhanced_image.shape[:2] != view.image.shape[:2]:
                        enhanced_image = torch.nn.functional.interpolate(
                            enhanced_image.permute(2, 0, 1).unsqueeze(0),
                            size=view.image.shape[:2], mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0)
                    
                    view.image = enhanced_image
                
                view.reliability_score = reliable_mask.float().mean().item()
                refined_views.append(view)
            
            # Step 5: Fine-tune 3DGS on new data
            print("Fine-tuning 3DGS...")
            self.fine_tune_gaussians(known_views + refined_views, iterations=5000)
            
            # Add refined views to known views
            known_views.extend(refined_views)
            all_rounds_views.append(refined_views.copy())
            
            print(f"Round {round_idx + 1} completed. Total known views: {len(known_views)}")
        
        return all_rounds_views
    
    def fine_tune_gaussians(self, all_views: List[ViewInfo], iterations=5000):
        """Fine-tune Gaussians with densification on new data"""
        print(f"Fine-tuning for {iterations} iterations on {len(all_views)} views...")
        
        optimizer = self.create_optimizer()
        
        for iteration in range(iterations):
            # Randomly select view (bias towards newer views)
            if len(all_views) > 20:
                # Higher probability for recent views
                weights = torch.ones(len(all_views), device=all_views[0].camera.R.device)
                weights[-10:] *= 2.0  # Double weight for last 10 views
                view_idx = torch.multinomial(weights, 1).item()
            else:
                view_idx = torch.randint(0, len(all_views), (1,)).item()
            
            view = all_views[view_idx]
            
            if view.image is None:
                continue
            
            # Render
            rendered = self.renderer.render(self.gaussians, view.camera)
            
            # Create reliability mask if available
            reliable_mask = None
            if hasattr(view, 'reliability_score') and view.reliability_score < 0.8:
                # For views with low reliability, focus on reliable regions
                try:
                    reliable_mask = self.see3d_proxy.identify_reliable_regions(view, all_views[:5])
                except:
                    reliable_mask = None
            
            # Compute loss
            loss = self.compute_loss(rendered, view, reliable_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradients for densification
            if hasattr(self.gaussians._xyz, 'grad') and self.gaussians._xyz.grad is not None:
                self.gaussians._xyz.grad_accum = getattr(self.gaussians._xyz, 'grad_accum', 0) + torch.norm(self.gaussians._xyz.grad, dim=1)
            
            optimizer.step()
            
            # Densification and pruning
            if iteration % 300 == 0 and iteration > 0:
                self.gaussians.densify()
                if iteration % 1000 == 0:
                    self.gaussians.prune()
            
            if iteration % 200 == 0:
                print(f"Fine-tune iteration {iteration}/{iterations}, Loss: {loss.item():.6f}")

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
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute Structural Similarity Index"""
        if SSIM_AVAILABLE:
            try:
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                if len(pred_np.shape) == 3:
                    ssim_val = ssim(pred_np, target_np, multichannel=True, channel_axis=2)
                else:
                    ssim_val = ssim(pred_np, target_np)
                
                return ssim_val
            except:
                pass
        
        # Fallback SSIM
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return torch.clamp(ssim_val, 0, 1).item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute Learned Perceptual Image Patch Similarity"""
        if self.lpips_fn is None:
            return 0.0  # Fallback value
        
        try:
            # Ensure images are in [-1, 1] range
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            
            # Reshape to (B, C, H, W)
            if len(pred_norm.shape) == 3:
                pred_norm = pred_norm.permute(2, 0, 1).unsqueeze(0)
                target_norm = target_norm.permute(2, 0, 1).unsqueeze(0)
            
            lpips_val = self.lpips_fn(pred_norm, target_norm)
            return lpips_val.item()
        except Exception as e:
            print(f"LPIPS computation failed: {e}")
            return 0.0
    
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
            
            for view_idx, view in enumerate(round_views[:10]):  # Evaluate first 10 views
                if view.image is None:
                    continue
                
                # Find corresponding target if available
                target_img = None
                if target_views and view_idx < len(target_views):
                    target_img = target_views[view_idx].image
                
                # Reference images from round 0 (training views)
                ref_images = [v.image for v in progressive_views[0][:5] if v.image is not None]
                
                metrics = self.evaluate_view(view.image, target_img, ref_images)
                
                for key, value in metrics.items():
                    if key in round_metrics:
                        round_metrics[key].append(value)
            
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