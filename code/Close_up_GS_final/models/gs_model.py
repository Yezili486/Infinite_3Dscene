"""
Initial Gaussian Splatting Model based on 2DGS
Implementation of the baseline model from the paper (Section 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from utils.camera import Camera


@dataclass
class GaussianPrimitive:
    """
    Gaussian primitive containing all necessary parameters
    """
    center: torch.Tensor        # [N, 3] - Gaussian centers
    covariance: torch.Tensor    # [N, 3, 3] - Covariance matrices
    opacity: torch.Tensor       # [N, 1] - Opacity values
    sh_coeffs: torch.Tensor     # [N, (SH_degree+1)^2, 3] - Spherical harmonics coefficients
    scales: torch.Tensor        # [N, 3] - Scaling factors
    rotations: torch.Tensor     # [N, 4] - Quaternion rotations


class GSModel(nn.Module):
    """
    Initial Gaussian Splatting Model (Baseline)
    Based on 2DGS with modifications for Close-up view synthesis
    """
    
    def __init__(self, config):
        super(GSModel, self).__init__()
        self.config = config
        
        # Model parameters
        self.max_gaussians = config.get('max_gaussians', 100000)
        self.sh_degree = config.get('sh_degree', 3)
        self.densify_threshold = config.get('densify_threshold', 1/3)  # Distance < 1/3
        
        # Initialize Gaussian parameters
        self._init_gaussian_parameters()
        
        # Activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize
        
        # Training state
        self.training_iteration = 0
        self.densify_from_iter = config.get('densify_from_iter', 500)
        self.densify_until_iter = config.get('densify_until_iter', 15000)
        self.densification_interval = config.get('densification_interval', 100)
        
        # Loss function parameters
        self.lambda_ssim = config.get('lambda_ssim', 0.2)
        
        # Initialize SSIM loss
        self.ssim_loss = SSIMLoss()
    
    def _init_gaussian_parameters(self):
        """Initialize Gaussian parameters"""
        # Initialize with empty tensors, will be populated during training
        self._centers = nn.Parameter(torch.empty(0, 3))
        self._scales = nn.Parameter(torch.empty(0, 3))
        self._rotations = nn.Parameter(torch.empty(0, 4))
        self._opacities = nn.Parameter(torch.empty(0, 1))
        
        # Spherical harmonics coefficients
        # (SH_degree + 1)^2 coefficients for each color channel
        num_sh_coeffs = (self.sh_degree + 1) ** 2
        self._sh_coeffs = nn.Parameter(torch.empty(0, num_sh_coeffs, 3))
        
        # Additional parameters for training
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
    
    def create_from_point_cloud(self, points: torch.Tensor, colors: torch.Tensor):
        """
        Initialize Gaussians from point cloud
        
        Args:
            points: Point positions [N, 3]
            colors: Point colors [N, 3]
        """
        num_points = points.shape[0]
        print(f"Initializing {num_points} Gaussians from point cloud")
        
        # Get device from input points
        device = points.device
        
        # Initialize centers
        centers = points.clone()
        
        # Initialize scales (small initial size)
        scales = torch.log(torch.ones(num_points, 3, device=device) * 0.01)
        
        # Initialize rotations (identity quaternions)
        rotations = torch.zeros(num_points, 4, device=device)
        rotations[:, 0] = 1.0  # w component
        
        # Initialize opacities (logit of 0.1)
        opacities = torch.logit(torch.ones(num_points, 1, device=device) * 0.1)
        
        # Initialize SH coefficients
        num_sh_coeffs = (self.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(num_points, num_sh_coeffs, 3, device=device)
        
        # Set DC component (first coefficient) to the color
        sh_coeffs[:, 0, :] = colors.clone()
        
        # Set parameters
        self._centers = nn.Parameter(centers.requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self._rotations = nn.Parameter(rotations.requires_grad_(True))
        self._opacities = nn.Parameter(opacities.requires_grad_(True))
        self._sh_coeffs = nn.Parameter(sh_coeffs.requires_grad_(True))
        
        # Initialize training-related tensors
        self.max_radii2D = torch.zeros(num_points, device=device)
        self.xyz_gradient_accum = torch.zeros(num_points, 1, device=device)
        self.denom = torch.zeros(num_points, 1, device=device)
    
    @property
    def get_centers(self) -> torch.Tensor:
        """Get Gaussian centers"""
        return self._centers
    
    @property
    def get_scales(self) -> torch.Tensor:
        """Get activated scales"""
        return self.scaling_activation(self._scales)
    
    @property
    def get_rotations(self) -> torch.Tensor:
        """Get normalized rotations"""
        return self.rotation_activation(self._rotations, dim=-1)
    
    @property
    def get_opacities(self) -> torch.Tensor:
        """Get activated opacities"""
        return self.opacity_activation(self._opacities)
    
    @property
    def get_sh_coeffs(self) -> torch.Tensor:
        """Get SH coefficients"""
        return self._sh_coeffs
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scales and rotations
        
        Returns:
            Covariance matrices [N, 3, 3]
        """
        scales = self.get_scales
        rotations = self.get_rotations
        
        # Build scaling matrix
        S = torch.zeros(scales.shape[0], 3, 3, device=scales.device)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]
        
        # Build rotation matrix from quaternion
        R = quaternion_to_rotation_matrix(rotations)
        
        # Covariance = R * S * S^T * R^T
        RS = torch.bmm(R, S)
        covariance = torch.bmm(RS, RS.transpose(-2, -1))
        
        return covariance
    
    def forward(self, camera: Camera, training_views_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward rendering function (Equations 1-3 from paper Section 3)
        Memory optimized with AMP for RTX 3070Ti (8GB)
        
        Args:
            camera: Camera object for rendering
            training_views_mask: Mask for training views (optional)
            
        Returns:
            Dictionary containing rendered outputs
        """
        # Use mixed precision for memory efficiency on RTX 3070Ti
        with torch.cuda.amp.autocast():
            # Get Gaussian parameters
            centers = self.get_centers  # μ in equation (1)
            covariances = self.get_covariance_matrices()  # Σ in equation (1)
            opacities = self.get_opacities  # α in equation (2)
            sh_coeffs = self.get_sh_coeffs
            
            # Render image using Gaussian Splatting
            rendered_image, depth_map, weights = self.render_gaussians(
                centers, covariances, opacities, sh_coeffs, camera
            )
        
        return {
            'image': rendered_image,
            'depth': depth_map,
            'weights': weights,
            'num_gaussians': centers.shape[0]
        }
    
    def render_gaussians(self, centers: torch.Tensor, covariances: torch.Tensor, 
                        opacities: torch.Tensor, sh_coeffs: torch.Tensor, 
                        camera: Camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to image (Implementation of equations 1-3)
        
        Args:
            centers: Gaussian centers [N, 3]
            covariances: Covariance matrices [N, 3, 3]
            opacities: Opacity values [N, 1]
            sh_coeffs: SH coefficients [N, K, 3]
            camera: Camera for rendering
            
        Returns:
            rendered_image: [3, H, W]
            depth_map: [1, H, W]
            weights: [N] - blending weights for each Gaussian
        """
        height, width = camera.image_height, camera.image_width
        
        # Project Gaussians to 2D (Equation 1)
        projected_means, projected_covariances, depths = self.project_gaussians_to_2d(
            centers, covariances, camera
        )
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=centers.device),
            torch.arange(width, dtype=torch.float32, device=centers.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        
        # Initialize output tensors
        rendered_image = torch.zeros(3, height, width, device=centers.device)
        depth_map = torch.zeros(1, height, width, device=centers.device)
        
        # Alpha blending (Equation 2)
        # For each pixel, blend contributions from all Gaussians
        accumulated_alpha = torch.zeros(height, width, device=centers.device)
        weights = torch.zeros(centers.shape[0], device=centers.device)
        
        # Sort Gaussians by depth (back to front)
        depth_indices = torch.argsort(depths)
        
        for i in depth_indices:
            # Compute 2D Gaussian weight for all pixels (Equation 1)
            mean_2d = projected_means[i]  # [2]
            cov_2d = projected_covariances[i]  # [2, 2]
            
            # Compute inverse covariance
            det = cov_2d[0, 0] * cov_2d[1, 1] - cov_2d[0, 1] * cov_2d[1, 0]
            if det <= 0:
                continue
                
            inv_cov = torch.zeros_like(cov_2d)
            inv_cov[0, 0] = cov_2d[1, 1] / det
            inv_cov[1, 1] = cov_2d[0, 0] / det
            inv_cov[0, 1] = inv_cov[1, 0] = -cov_2d[0, 1] / det
            
            # Compute Gaussian weights for all pixels
            diff = pixel_coords - mean_2d  # [H, W, 2]
            
            # Mahalanobis distance: (x-μ)^T Σ^-1 (x-μ)
            mahal_dist = torch.sum(diff.unsqueeze(-2) @ inv_cov.unsqueeze(0).unsqueeze(0) * diff.unsqueeze(-2), dim=-1)
            gaussian_weight = torch.exp(-0.5 * mahal_dist.squeeze(-1))  # [H, W]
            
            # Apply opacity
            alpha_i = opacities[i] * gaussian_weight  # [H, W]
            
            # Compute color from SH coefficients
            color_i = self.evaluate_sh_at_direction(sh_coeffs[i], camera, centers[i])  # [3]
            
            # Alpha blending (Equation 2)
            transmission = 1.0 - accumulated_alpha  # T_i in equation (2)
            contribution = alpha_i * transmission  # α_i * T_i
            
            # Update rendered image
            rendered_image += contribution.unsqueeze(0) * color_i.unsqueeze(-1).unsqueeze(-1)
            
            # Update depth map
            depth_map[0] += contribution * depths[i]
            
            # Update accumulated alpha
            accumulated_alpha += contribution
            
            # Store weight for this Gaussian
            weights[i] = contribution.sum()
            
            # Early termination if accumulated alpha is close to 1
            if accumulated_alpha.mean() > 0.99:
                break
        
        return rendered_image, depth_map, weights
    
    def project_gaussians_to_2d(self, centers: torch.Tensor, covariances: torch.Tensor, 
                               camera: Camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D Gaussians to 2D screen space
        
        Args:
            centers: Gaussian centers [N, 3]
            covariances: 3D covariance matrices [N, 3, 3]
            camera: Camera object
            
        Returns:
            projected_means: 2D projected centers [N, 2]
            projected_covariances: 2D covariance matrices [N, 2, 2]
            depths: Depths of Gaussians [N]
        """
        # Transform centers to camera space
        centers_homo = torch.cat([centers, torch.ones(centers.shape[0], 1, device=centers.device)], dim=1)
        world_to_camera = camera.world_to_camera.to(centers.device)
        centers_cam = (world_to_camera @ centers_homo.T).T[:, :3]
        
        # Project to screen space
        depths = centers_cam[:, 2]
        projected_means = camera.project_points(centers)
        
        # Project covariance matrices to 2D
        # This is a simplified projection - in practice, you'd use the Jacobian
        W = camera.world_to_camera[:3, :3].to(centers.device)
        projected_covariances = torch.zeros(centers.shape[0], 2, 2, device=centers.device)
        
        for i in range(centers.shape[0]):
            # Transform covariance to camera space
            cov_cam = W @ covariances[i] @ W.T
            
            # Project to 2D (simplified)
            if depths[i] > 0:
                focal_x, focal_y = camera.fx, camera.fy
                proj_scale = torch.tensor([[focal_x / depths[i], 0], 
                                         [0, focal_y / depths[i]]], device=centers.device)
                projected_covariances[i] = proj_scale @ cov_cam[:2, :2] @ proj_scale.T
            else:
                # Handle invalid depths
                projected_covariances[i] = torch.eye(2, device=centers.device) * 1e-6
        
        return projected_means, projected_covariances, depths
    
    def evaluate_sh_at_direction(self, sh_coeffs: torch.Tensor, camera: Camera, 
                                position: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics for view-dependent color
        
        Args:
            sh_coeffs: SH coefficients [K, 3]
            camera: Camera object
            position: 3D position [3]
            
        Returns:
            RGB color [3]
        """
        # Compute view direction
        camera_center = camera.camera_center.to(position.device)
        view_dir = camera_center - position
        view_dir = F.normalize(view_dir, dim=0)
        
        # Evaluate SH basis functions
        sh_values = evaluate_sh_basis(self.sh_degree, view_dir)  # [K]
        
        # Compute color
        color = torch.sum(sh_coeffs * sh_values.unsqueeze(-1), dim=0)  # [3]
        
        # Apply sigmoid to ensure valid color range
        color = torch.sigmoid(color)
        
        return color
    
    def compute_loss(self, rendered_output: Dict[str, torch.Tensor], 
                    target_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss function (Equation 4): L1 + λ SSIM
        
        Args:
            rendered_output: Output from forward pass
            target_image: Ground truth image [3, H, W]
            
        Returns:
            Dictionary containing loss components
        """
        rendered_image = rendered_output['image']
        
        # L1 loss
        l1_loss = F.l1_loss(rendered_image, target_image)
        
        # SSIM loss
        ssim_loss = self.ssim_loss(rendered_image.unsqueeze(0), target_image.unsqueeze(0))
        
        # Combined loss (Equation 4)
        total_loss = (1.0 - self.lambda_ssim) * l1_loss + self.lambda_ssim * ssim_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'num_gaussians': rendered_output['num_gaussians']
        }
    
    def densify(self, camera_distance: float):
        """
        Densification function (only when distance < 1/3, Paper Section 4.3.4)
        
        Args:
            camera_distance: Distance from camera to scene center
        """
        if camera_distance >= self.densify_threshold:
            return
        
        if (self.training_iteration < self.densify_from_iter or 
            self.training_iteration > self.densify_until_iter or
            self.training_iteration % self.densification_interval != 0):
            return
        
        print(f"Densifying at iteration {self.training_iteration} (distance: {camera_distance:.3f})")
        
        # Get current parameters
        centers = self.get_centers
        scales = self.get_scales
        rotations = self.get_rotations
        opacities = self.get_opacities
        sh_coeffs = self.get_sh_coeffs
        
        # Find Gaussians to densify based on gradients
        grad_threshold = 0.0002
        if self.xyz_gradient_accum.numel() > 0:
            gradients = self.xyz_gradient_accum / self.denom
            valid_mask = (gradients > grad_threshold).squeeze()
            
            if valid_mask.sum() > 0:
                # Clone high-gradient Gaussians
                new_centers = centers[valid_mask]
                new_scales = self._scales[valid_mask]
                new_rotations = self._rotations[valid_mask]
                new_opacities = self._opacities[valid_mask]
                new_sh_coeffs = sh_coeffs[valid_mask]
                
                # Add small random offset to new centers
                offset = torch.randn_like(new_centers) * 0.01
                new_centers = new_centers + offset
                
                # Concatenate with existing parameters
                self._centers = nn.Parameter(torch.cat([centers, new_centers], dim=0))
                self._scales = nn.Parameter(torch.cat([self._scales, new_scales], dim=0))
                self._rotations = nn.Parameter(torch.cat([self._rotations, new_rotations], dim=0))
                self._opacities = nn.Parameter(torch.cat([self._opacities, new_opacities], dim=0))
                self._sh_coeffs = nn.Parameter(torch.cat([self._sh_coeffs, new_sh_coeffs], dim=0))
                
                # Update auxiliary tensors
                num_new = new_centers.shape[0]
                self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(num_new)])
                self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(num_new, 1)])
                self.denom = torch.cat([self.denom, torch.zeros(num_new, 1)])
                
                print(f"Added {num_new} new Gaussians (total: {self._centers.shape[0]})")
    
    def prune_gaussians(self, min_opacity: float = 0.005):
        """Prune Gaussians with low opacity"""
        opacities = self.get_opacities
        valid_mask = (opacities > min_opacity).squeeze()
        
        if valid_mask.sum() < opacities.shape[0]:
            num_pruned = opacities.shape[0] - valid_mask.sum()
            print(f"Pruning {num_pruned} low-opacity Gaussians")
            
            # Keep only valid Gaussians
            self._centers = nn.Parameter(self._centers[valid_mask])
            self._scales = nn.Parameter(self._scales[valid_mask])
            self._rotations = nn.Parameter(self._rotations[valid_mask])
            self._opacities = nn.Parameter(self._opacities[valid_mask])
            self._sh_coeffs = nn.Parameter(self._sh_coeffs[valid_mask])
            
            # Update auxiliary tensors
            if self.max_radii2D.numel() > 0:
                self.max_radii2D = self.max_radii2D[valid_mask]
            if self.xyz_gradient_accum.numel() > 0:
                self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
            if self.denom.numel() > 0:
                self.denom = self.denom[valid_mask]
    
    def update_training_stats(self, gradients: torch.Tensor):
        """Update training statistics for densification"""
        if gradients is not None and gradients.shape[0] == self._centers.shape[0]:
            # Update gradient accumulation
            grad_norm = torch.norm(gradients, dim=1, keepdim=True)
            self.xyz_gradient_accum += grad_norm
            self.denom += 1.0
        
        self.training_iteration += 1
    
    def reset_opacity(self):
        """Reset opacity for all Gaussians"""
        with torch.no_grad():
            self._opacities.data = torch.logit(torch.ones_like(self._opacities) * 0.01)


class SSIMLoss(nn.Module):
    """SSIM Loss implementation"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))
    
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM"""
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.float()
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss"""
        # Ensure float32 type
        img1 = img1.float()
        img2 = img2.float()
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        channels = img1.shape[1]
        window = self.window.repeat(channels, 1, 1, 1).to(img1.device).float()
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1**2, window, padding=self.window_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2**2, window, padding=self.window_size//2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channels) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1.0 - ssim_map.mean()


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices
    
    Args:
        quaternions: [N, 4] quaternions (w, x, y, z)
        
    Returns:
        Rotation matrices [N, 3, 3]
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Normalize quaternions
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrices
    R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
    
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)
    
    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)
    
    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    
    return R


def evaluate_sh_basis(degree: int, direction: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics basis functions
    
    Args:
        degree: SH degree
        direction: Direction vector [3]
        
    Returns:
        SH basis values [(degree+1)^2]
    """
    x, y, z = direction[0], direction[1], direction[2]
    
    # Start with DC component
    sh_values = [0.282095]  # Y_0^0
    
    if degree >= 1:
        # Linear terms
        sh_values.extend([
            0.488603 * y,      # Y_1^{-1}
            0.488603 * z,      # Y_1^0
            0.488603 * x       # Y_1^1
        ])
    
    if degree >= 2:
        # Quadratic terms
        sh_values.extend([
            1.092548 * x * y,                    # Y_2^{-2}
            1.092548 * y * z,                    # Y_2^{-1}
            0.315392 * (3*z*z - 1),            # Y_2^0
            1.092548 * x * z,                    # Y_2^1
            0.546274 * (x*x - y*y)              # Y_2^2
        ])
    
    if degree >= 3:
        # Cubic terms (simplified)
        sh_values.extend([
            0.590044 * y * (3*x*x - y*y),      # Y_3^{-3}
            2.890611 * x * y * z,              # Y_3^{-2}
            0.457046 * y * (5*z*z - 1),        # Y_3^{-1}
            0.373176 * z * (5*z*z - 3),        # Y_3^0
            0.457046 * x * (5*z*z - 1),        # Y_3^1
            1.445306 * z * (x*x - y*y),        # Y_3^2
            0.590044 * x * (x*x - 3*y*y)       # Y_3^3
        ])
    
    return torch.tensor(sh_values[:((degree+1)**2)], device=direction.device)


if __name__ == '__main__':
    # Test GSModel
    from utils.config import Config
    
    config = Config()
    model = GSModel(config)
    
    # Test with dummy point cloud
    points = torch.randn(1000, 3)
    colors = torch.rand(1000, 3)
    
    model.create_from_point_cloud(points, colors)
    
    print(f"Initialized model with {model.get_centers.shape[0]} Gaussians")
    print("GSModel implementation completed!")
