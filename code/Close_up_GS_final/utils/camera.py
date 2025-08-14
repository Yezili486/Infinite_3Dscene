"""
Camera utilities for Close-up-GS
"""

import torch
import numpy as np
from typing import Tuple, Optional
import json
from pathlib import Path


class Camera:
    """Camera class for view synthesis"""
    
    def __init__(self, 
                 image_width: int,
                 image_height: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 world_to_camera: torch.Tensor,
                 camera_to_world: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize camera
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            fx, fy: Focal lengths
            cx, cy: Principal point
            world_to_camera: World to camera transformation matrix [4, 4]
            camera_to_world: Camera to world transformation matrix [4, 4]
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Determine device
        if device is not None:
            self.device = device
        else:
            self.device = world_to_camera.device
        
        # Move matrices to correct device
        self.world_to_camera = world_to_camera.to(self.device)
        if camera_to_world is not None:
            self.camera_to_world = camera_to_world.to(self.device)
        else:
            self.camera_to_world = torch.inverse(self.world_to_camera)
        
        # Extract camera center from camera_to_world matrix
        self.camera_center = self.camera_to_world[:3, 3]
        
        # Compute projection matrix
        self.projection_matrix = self._compute_projection_matrix()
        
        # Compute full transformation matrix
        self.full_proj_transform = self.camera_to_world @ self.projection_matrix
    
    def _compute_projection_matrix(self) -> torch.Tensor:
        """Compute OpenGL-style projection matrix"""
        # Near and far planes
        znear = 0.01
        zfar = 100.0
        
        # Compute projection matrix
        proj = torch.zeros(4, 4, device=self.device)
        proj[0, 0] = 2.0 * self.fx / self.image_width
        proj[1, 1] = 2.0 * self.fy / self.image_height
        proj[0, 2] = (self.image_width - 2.0 * self.cx) / self.image_width
        proj[1, 2] = (self.image_height - 2.0 * self.cy) / self.image_height
        proj[2, 2] = -(zfar + znear) / (zfar - znear)
        proj[2, 3] = -2.0 * zfar * znear / (zfar - znear)
        proj[3, 2] = -1.0
        
        return proj
    
    def to(self, device: torch.device) -> 'Camera':
        """Move camera to specified device"""
        if device == self.device:
            return self
        
        self.device = device
        self.world_to_camera = self.world_to_camera.to(device)
        self.camera_to_world = self.camera_to_world.to(device)
        self.camera_center = self.camera_center.to(device)
        return self
    
    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        """Get camera intrinsic matrix"""
        K = torch.zeros(3, 3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K
    
    def project_points(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to 2D image plane
        
        Args:
            points_3d: 3D points [N, 3]
            
        Returns:
            2D points [N, 2]
        """
        # Transform to camera coordinates
        points_homo = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)], dim=1)
        world_to_camera = self.world_to_camera.to(points_3d.device)
        
        points_cam = (world_to_camera @ points_homo.T).T[:, :3]
        
        # Project to image plane
        x_proj = points_cam[:, 0] / points_cam[:, 2]
        y_proj = points_cam[:, 1] / points_cam[:, 2]
        
        # Apply intrinsics
        u = self.fx * x_proj + self.cx
        v = self.fy * y_proj + self.cy
        
        return torch.stack([u, v], dim=1)
    
    def unproject_points(self, points_2d: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Unproject 2D points to 3D space
        
        Args:
            points_2d: 2D points [N, 2]
            depths: Corresponding depths [N]
            
        Returns:
            3D points [N, 3]
        """
        u, v = points_2d[:, 0], points_2d[:, 1]
        
        # Convert to normalized camera coordinates
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        # Scale by depth
        x_cam = x_norm * depths
        y_cam = y_norm * depths
        z_cam = depths
        
        # Transform to world coordinates
        points_cam_homo = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(depths)], dim=1)
        camera_to_world = self.camera_to_world.to(points_cam_homo.device)  # Ensure on device
        
        points_world = (camera_to_world @ points_cam_homo.T).T[:, :3]
        
        return points_world
    
    def get_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ray origins and directions for all pixels
        
        Returns:
            ray_origins: Ray origins [H, W, 3]
            ray_directions: Ray directions [H, W, 3]
        """
        # Create pixel grid
        i, j = torch.meshgrid(
            torch.arange(self.image_width, dtype=torch.float32),
            torch.arange(self.image_height, dtype=torch.float32),
            indexing='xy'
        )
        
        # Convert to normalized camera coordinates
        dirs = torch.stack([
            (i - self.cx) / self.fx,
            -(j - self.cy) / self.fy,  # Flip y-axis
            -torch.ones_like(i)
        ], dim=-1)
        
        # Transform directions to world space
        rays_d = dirs @ self.camera_to_world[:3, :3].T
        
        # Ray origins are camera center
        rays_o = self.camera_center.expand(rays_d.shape)
        
        return rays_o, rays_d


def load_cameras_from_transforms(transforms_path: str) -> list:
    """
    Load cameras from transforms.json file (NeRF format)
    
    Args:
        transforms_path: Path to transforms.json file
        
    Returns:
        List of Camera objects
    """
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    cameras = []
    
    # Get camera intrinsics
    if 'camera_angle_x' in transforms:
        camera_angle_x = transforms['camera_angle_x']
        w = transforms.get('w', 800)
        h = transforms.get('h', 600)
        fx = fy = 0.5 * w / np.tan(0.5 * camera_angle_x)
        cx = w / 2.0
        cy = h / 2.0
    else:
        # Use provided intrinsics
        fx = transforms.get('fl_x', 800.0)
        fy = transforms.get('fl_y', 800.0)
        cx = transforms.get('cx', 400.0)
        cy = transforms.get('cy', 300.0)
        w = transforms.get('w', 800)
        h = transforms.get('h', 600)
    
    # Load frames
    for frame in transforms['frames']:
        # Get transformation matrix
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
        # Convert from NeRF to OpenGL coordinate system
        # NeRF: +Y up, +Z forward, +X right
        # OpenGL: +Y up, -Z forward, +X right
        transform_matrix[:, 1:3] *= -1
        
        camera = Camera(
            image_width=int(w),
            image_height=int(h),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_world=transform_matrix
        )
        
        cameras.append(camera)
    
    return cameras


def create_spiral_cameras(center: torch.Tensor,
                         radius: float,
                         height: float,
                         num_views: int,
                         image_width: int = 800,
                         image_height: int = 600,
                         focal_length: float = 800.0) -> list:
    """
    Create cameras on a spiral path for novel view synthesis
    
    Args:
        center: Center point to look at [3]
        radius: Spiral radius
        height: Height variation
        num_views: Number of views to generate
        image_width: Image width
        image_height: Image height
        focal_length: Focal length
        
    Returns:
        List of Camera objects
    """
    cameras = []
    
    for i in range(num_views):
        # Spiral parameters
        angle = 2 * np.pi * i / num_views
        cam_height = height * np.sin(2 * np.pi * i / num_views)
        
        # Camera position
        cam_pos = torch.tensor([
            center[0] + radius * np.cos(angle),
            center[1] + cam_height,
            center[2] + radius * np.sin(angle)
        ])
        
        # Look-at transformation
        forward = center - cam_pos
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Create transformation matrix
        camera_to_world = torch.eye(4)
        camera_to_world[:3, 0] = right
        camera_to_world[:3, 1] = up
        camera_to_world[:3, 2] = -forward
        camera_to_world[:3, 3] = cam_pos
        
        camera = Camera(
            image_width=image_width,
            image_height=image_height,
            fx=focal_length,
            fy=focal_length,
            cx=image_width / 2.0,
            cy=image_height / 2.0,
            camera_to_world=camera_to_world
        )
        
        cameras.append(camera)
    
    return cameras


if __name__ == '__main__':
    # Test camera functionality
    
    # Create a test camera
    camera = Camera(
        image_width=800,
        image_height=600,
        fx=800.0,
        fy=800.0,
        cx=400.0,
        cy=300.0,
        world_to_camera=torch.eye(4)
    )
    
    # Test point projection
    points_3d = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    points_2d = camera.project_points(points_3d)
    print(f"3D points: {points_3d}")
    print(f"2D points: {points_2d}")
    
    # Test ray generation
    rays_o, rays_d = camera.get_rays()
    print(f"Ray origins shape: {rays_o.shape}")
    print(f"Ray directions shape: {rays_d.shape}")
    
    print("Camera utilities test completed!")
