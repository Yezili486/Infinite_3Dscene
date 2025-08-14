"""
Dataset utilities for Close-up-GS
Handles loading and preprocessing of training data
Supports LERF and LLFF datasets as specified in paper section 5.1
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import os
from scipy.spatial.transform import Rotation

from utils.camera import Camera, load_cameras_from_transforms
from utils.colmap_parser import COLMAPLoader


class CloseUpDataset(Dataset):
    """
    Dataset for Close-up Gaussian Splatting
    Supports LERF and LLFF datasets as specified in paper section 5.1
    """
    
    def __init__(self, 
                 data_path: str,
                 config,
                 split: str = 'train',
                 dataset_type: str = 'auto',  # 'lerf', 'llff', 'nerf', 'auto'
                 target_resolution: Tuple[int, int] = (512, 512)):
        """
        Initialize Close-up Dataset
        
        Args:
            data_path: Path to dataset directory
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            dataset_type: Type of dataset ('lerf', 'llff', 'nerf', 'auto')
            target_resolution: Target resolution for preprocessing (width, height)
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.target_resolution = target_resolution
        
        # Auto-detect dataset type if not specified
        if dataset_type == 'auto':
            self.dataset_type = self._detect_dataset_type()
        else:
            self.dataset_type = dataset_type
        
        print(f"Loading {self.dataset_type.upper()} dataset from {data_path}")
        
        # Load dataset based on type
        if self.dataset_type == 'lerf':
            self._load_lerf_dataset()
        elif self.dataset_type == 'llff':
            self._load_llff_dataset()
        elif self.dataset_type == 'nerf':
            self._load_nerf_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Extract training views (far distance) and object center
        self.training_views = self._extract_training_views()  # Vs (far distance views)
        self.object_center = self._compute_object_center()    # p_target
        
        # Extract close-up test views for evaluation
        self.closeup_test_views = self._extract_closeup_test_views()
        
        print(f"Loaded {len(self.cameras)} total cameras")
        print(f"Training views (far distance): {len(self.training_views)}")
        print(f"Close-up test views: {len(self.closeup_test_views)}")
        print(f"Object center: {self.object_center}")
        print(f"Target resolution: {self.target_resolution}")
    
    def _detect_dataset_type(self) -> str:
        """Auto-detect dataset type based on file structure"""
        # Check for LERF indicators
        if (self.data_path / "lerf_config.json").exists():
            return 'lerf'
        
        # Check for LLFF indicators
        if (self.data_path / "poses_bounds.npy").exists():
            return 'llff'
        
        # Check for NeRF indicators
        if any((self.data_path / f"transforms_{split}.json").exists() 
               for split in ['train', 'val', 'test']):
            return 'nerf'
        
        # Default to NeRF if transforms.json exists
        if (self.data_path / "transforms.json").exists():
            return 'nerf'
        
        raise ValueError("Cannot auto-detect dataset type. Please specify manually.")
    
    def _load_lerf_dataset(self):
        """Load LERF dataset format"""
        # LERF datasets typically have a config file and image directory
        config_file = self.data_path / "lerf_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                lerf_config = json.load(f)
        else:
            # Use default LERF structure
            lerf_config = {}
        
        # Load images and poses
        images_dir = self.data_path / "images"
        if not images_dir.exists():
            images_dir = self.data_path / "rgb"
        
        # Get image files
        image_files = sorted([f for f in images_dir.glob("*") 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        # Load poses (LERF typically uses COLMAP format)
        poses_file = self.data_path / "poses.txt"
        if poses_file.exists():
            self.cameras, self.images = self._load_colmap_poses_and_images(poses_file, image_files)
        else:
            # Try transforms.json format
            transforms_file = self.data_path / "transforms.json"
            if transforms_file.exists():
                self.cameras = load_cameras_from_transforms(str(transforms_file))
                self.images = self._load_images_from_transforms(transforms_file)
            else:
                raise FileNotFoundError("No pose file found for LERF dataset")
    
    def _load_llff_dataset(self):
        """Load LLFF dataset format"""
        # LLFF datasets use poses_bounds.npy format
        poses_bounds_file = self.data_path / "poses_bounds.npy"
        
        if not poses_bounds_file.exists():
            raise FileNotFoundError("poses_bounds.npy not found for LLFF dataset")
        
        # Load poses and bounds
        poses_bounds = np.load(poses_bounds_file)
        poses = poses_bounds[:, :-2].reshape(-1, 3, 5)  # [N, 3, 5]
        bounds = poses_bounds[:, -2:]  # [N, 2]
        
        # Load images
        images_dir = self.data_path / "images"
        if not images_dir.exists():
            images_dir = self.data_path / "images_4" if (self.data_path / "images_4").exists() else self.data_path
        
        image_files = sorted([f for f in images_dir.glob("*") 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        # Convert LLFF format to our camera format
        self.cameras, self.images = self._convert_llff_to_cameras(poses, bounds, image_files)
    
    def _load_nerf_dataset(self):
        """Load NeRF dataset format"""
        # Try split-specific transform files first
        transforms_file = self.data_path / f"transforms_{self.split}.json"
        if not transforms_file.exists():
            transforms_file = self.data_path / "transforms.json"
        
        if not transforms_file.exists():
            raise FileNotFoundError(f"No transforms file found for NeRF dataset")
        
        self.cameras = load_cameras_from_transforms(str(transforms_file))
        self.images = self._load_images_from_transforms(transforms_file)
    
    def _load_images_from_transforms(self, transforms_file: Path) -> List[torch.Tensor]:
        """Load images based on transforms.json file"""
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        images = []
        for frame in transforms['frames']:
            image_path = self.data_path / frame['file_path']
            
            # Handle different extensions
            if not image_path.exists():
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = image_path.with_suffix(ext)
                    if test_path.exists():
                        image_path = test_path
                        break
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image = self._load_and_preprocess_image(image_path)
            images.append(image)
        
        return images
    
    def _load_colmap_poses_and_images(self, poses_file: Path, image_files: List[Path]) -> Tuple[List[Camera], List[torch.Tensor]]:
        """Load COLMAP format poses and corresponding images"""
        try:
            # Try to load COLMAP reconstruction from parent directory
            colmap_dir = poses_file.parent
            
            # Look for COLMAP files
            if not any(f.exists() for f in [colmap_dir / "cameras.bin", colmap_dir / "cameras.txt"]):
                # Fall back to simple parsing if no COLMAP files found
                return self._load_simple_poses_and_images(poses_file, image_files)
            
            # Use proper COLMAP loader
            colmap_loader = COLMAPLoader(colmap_dir, images_path=colmap_dir / "images")
            cameras, images = colmap_loader.get_cameras_and_images(self.target_resolution)
            
            return cameras, images
            
        except Exception as e:
            print(f"Warning: Failed to load COLMAP data: {e}")
            print("Falling back to simple pose loading...")
            return self._load_simple_poses_and_images(poses_file, image_files)
    
    def _load_simple_poses_and_images(self, poses_file: Path, image_files: List[Path]) -> Tuple[List[Camera], List[torch.Tensor]]:
        """Simple fallback pose loading"""
        cameras = []
        images = []
        
        for i, image_file in enumerate(image_files):
            # Create simple camera with default parameters
            camera = Camera(
                image_width=self.target_resolution[0],
                image_height=self.target_resolution[1],
                fx=800.0, fy=800.0,
                cx=self.target_resolution[0]/2, cy=self.target_resolution[1]/2,
                world_to_camera=torch.eye(4)
            )
            
            image = self._load_and_preprocess_image(image_file)
            
            cameras.append(camera)
            images.append(image)
        
        return cameras, images
    
    def _convert_llff_to_cameras(self, poses: np.ndarray, bounds: np.ndarray, image_files: List[Path]) -> Tuple[List[Camera], List[torch.Tensor]]:
        """Convert LLFF poses to Camera objects"""
        cameras = []
        images = []
        
        for i, (pose, bound, image_file) in enumerate(zip(poses, bounds, image_files)):
            # Extract camera parameters from LLFF format
            # pose format: [down, right, back, position, hwf]
            hwf = pose[:, -1]  # [height, width, focal_length]
            height, width, focal_length = hwf
            
            # Camera intrinsics
            fx = fy = focal_length
            cx = width / 2.0
            cy = height / 2.0
            
            # Camera extrinsics (convert LLFF coordinate system)
            c2w = pose[:, :4]  # [3, 4]
            c2w_homogeneous = np.eye(4)
            c2w_homogeneous[:3, :] = c2w
            
            # Convert from LLFF to OpenGL coordinate system
            c2w_homogeneous[:3, 1:3] *= -1  # Flip Y and Z axes
            
            camera = Camera(
                image_width=int(width),
                image_height=int(height),
                fx=fx, fy=fy, cx=cx, cy=cy,
                camera_to_world=torch.from_numpy(c2w_homogeneous).float()
            )
            
            image = self._load_and_preprocess_image(image_file)
            
            cameras.append(camera)
            images.append(image)
        
        return cameras, images
    
    def _load_and_preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess image with resize to 512x512 to avoid channel mismatch
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Resize to target resolution (512x512) to avoid channel mismatch
        image = cv2.resize(image, self.target_resolution, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        
        return image_tensor
    
    def _extract_training_views(self) -> List[int]:
        """
        Extract far distance training views (Vs)
        Returns indices of cameras that are far from the object center
        """
        if not hasattr(self, 'cameras') or len(self.cameras) == 0:
            return []
        
        # Compute distances from object center
        distances = []
        for camera in self.cameras:
            distance = torch.norm(camera.camera_center - self.object_center)
            distances.append(distance.item())
        
        # Select views that are in the far distance range (top 70% by distance)
        sorted_indices = np.argsort(distances)
        num_far_views = int(len(sorted_indices) * 0.7)
        training_indices = sorted_indices[-num_far_views:].tolist()
        
        return training_indices
    
    def _compute_object_center(self) -> torch.Tensor:
        """
        Compute object center (p_target) from camera positions
        """
        if not hasattr(self, 'cameras') or len(self.cameras) == 0:
            return torch.zeros(3)
        
        # Simple method: average of all camera look-at points
        look_at_points = []
        for camera in self.cameras:
            # Compute look-at point (camera position + forward direction)
            forward = camera.camera_to_world[:3, 2]  # -Z axis in camera space
            look_at = camera.camera_center - forward * 3.0  # Assume looking at 3 units forward
            look_at_points.append(look_at)
        
        # Average all look-at points
        object_center = torch.stack(look_at_points).mean(dim=0)
        
        # 步骤3: 确保object_center在正确设备上
        if len(self.cameras) > 0:
            target_device = self.cameras[0].camera_center.device
            object_center = object_center.to(target_device)
        
        return object_center
    
    def _extract_closeup_test_views(self) -> List[int]:
        """
        Extract close-up test views for evaluation
        Returns indices of cameras that are close to the object center
        """
        if not hasattr(self, 'cameras') or len(self.cameras) == 0:
            return []
        
        # Compute distances from object center
        distances = []
        for camera in self.cameras:
            distance = torch.norm(camera.camera_center - self.object_center)
            distances.append(distance.item())
        
        # Select views that are in close-up range (bottom 30% by distance)
        sorted_indices = np.argsort(distances)
        num_closeup_views = int(len(sorted_indices) * 0.3)
        closeup_indices = sorted_indices[:num_closeup_views].tolist()
        
        return closeup_indices
    
    def warp_image(self, source_image: torch.Tensor, source_camera: Camera, 
                   target_camera: Camera) -> torch.Tensor:
        """
        Warp function from paper section 4.2
        Warp from reference view to novel view
        
        Args:
            source_image: Source image [C, H, W]
            source_camera: Source camera
            target_camera: Target camera
            
        Returns:
            Warped image [C, H, W]
        """
        C, H, W = source_image.shape
        
        # Create pixel grid for target camera
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing='xy'
        )
        
        # Convert to homogeneous coordinates
        pixels = torch.stack([i, j, torch.ones_like(i)], dim=-1)  # [H, W, 3]
        pixels = pixels.reshape(-1, 3)  # [H*W, 3]
        
        # Unproject to 3D using estimated depth (simplified)
        depths = torch.ones(pixels.shape[0]) * 3.0  # Assume depth of 3.0
        rays_3d = target_camera.unproject_points(pixels[:, :2], depths)
        
        # Project to source camera
        projected_2d = source_camera.project_points(rays_3d)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        grid_x = 2.0 * projected_2d[:, 0] / W - 1.0
        grid_y = 2.0 * projected_2d[:, 1] / H - 1.0
        
        # Reshape for grid_sample
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H, W, 2)
        
        # Sample from source image
        source_image_batch = source_image.unsqueeze(0)  # [1, C, H, W]
        warped_image = torch.nn.functional.grid_sample(
            source_image_batch, grid, 
            mode='bilinear', padding_mode='border', align_corners=False
        )
        
        return warped_image.squeeze(0)  # [C, H, W]
    
    def __len__(self) -> int:
        """Get dataset length"""
        if self.split == 'train':
            return len(self.training_views)
        else:
            return len(self.cameras)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing image, pose, focal_length, and metadata
        """
        # Select appropriate camera index based on split
        if self.split == 'train':
            camera_idx = self.training_views[idx] if idx < len(self.training_views) else idx
        else:
            camera_idx = idx
        
        camera = self.cameras[camera_idx]
        image = self.images[camera_idx]
        
        # Apply data augmentation for training
        if self.split == 'train':
            image = self._apply_augmentation(image)
        
        return {
            'image': image,  # [C, H, W] tensor
            'pose': camera.camera_to_world,  # [4, 4] camera pose matrix
            'focal_length': torch.tensor([camera.fx, camera.fy]),  # [2] focal lengths
            'camera': camera,  # Full camera object
            'idx': camera_idx,
            'is_closeup': camera_idx in self.closeup_test_views,
            'distance_to_center': torch.norm(camera.camera_center - self.object_center).item(),
            'object_center': self.object_center,  # p_target
            'image_width': camera.image_width,
            'image_height': camera.image_height
        }
    
    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation for training
        
        Args:
            image: Input image [C, H, W]
            
        Returns:
            Augmented image [C, H, W]
        """
        # Random brightness and contrast
        if random.random() < 0.5:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            image = torch.clamp(image * contrast + (brightness - 1.0), 0.0, 1.0)
        
        # Random color jittering (subtle for photorealistic data)
        if random.random() < 0.3:
            # Small color shift
            color_shift = torch.randn(3, 1, 1) * 0.02
            image = torch.clamp(image + color_shift, 0.0, 1.0)
        
        return image
    
    def get_closeup_test_samples(self) -> List[Dict]:
        """
        Get all close-up test samples for evaluation
        
        Returns:
            List of close-up test samples
        """
        closeup_samples = []
        for idx in self.closeup_test_views:
            camera = self.cameras[idx]
            image = self.images[idx]
            
            sample = {
                'image': image,
                'pose': camera.camera_to_world,
                'focal_length': torch.tensor([camera.fx, camera.fy]),
                'camera': camera,
                'idx': idx,
                'distance_to_center': torch.norm(camera.camera_center - self.object_center).item(),
                'object_center': self.object_center
            }
            closeup_samples.append(sample)
        
        return closeup_samples
    
    def get_training_views_info(self) -> Dict:
        """
        Get information about training views (Vs)
        
        Returns:
            Dictionary with training views statistics
        """
        if not self.training_views:
            return {}
        
        distances = []
        for idx in self.training_views:
            camera = self.cameras[idx]
            distance = torch.norm(camera.camera_center - self.object_center)
            distances.append(distance.item())
        
        return {
            'num_training_views': len(self.training_views),
            'min_distance': min(distances),
            'max_distance': max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'training_indices': self.training_views
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing and validation"""
    
    def __init__(self, 
                 num_samples: int = 100,
                 image_width: int = 800,
                 image_height: int = 600,
                 focal_length: float = 800.0):
        """
        Initialize synthetic dataset
        
        Args:
            num_samples: Number of synthetic samples to generate
            image_width: Image width
            image_height: Image height
            focal_length: Camera focal length
        """
        self.num_samples = num_samples
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        
        # Object center for view selection - 步骤3: 确保在CUDA设备上
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.object_center = torch.tensor([0., 0., 0.], device=device)  # Synthetic object at origin
        
        # Generate synthetic cameras and images
        self.cameras, self.images = self._generate_synthetic_data()
        
        # Training views
        self.training_views = list(range(min(20, len(self.cameras))))
    
    def _generate_synthetic_data(self) -> Tuple[List[Camera], List[torch.Tensor]]:
        """Generate synthetic cameras and images"""
        cameras = []
        images = []
        
        for i in range(self.num_samples):
            # Random camera position on sphere
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            radius = random.uniform(2.0, 5.0)
            
            cam_pos = torch.tensor([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.cos(phi),
                radius * np.sin(phi) * np.sin(theta)
            ], dtype=torch.float32)
            
            # Look at origin
            forward = -cam_pos / torch.norm(cam_pos)
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            right = torch.cross(forward, up)
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            
            # Create camera
            camera_to_world = torch.eye(4)
            camera_to_world[:3, 0] = right
            camera_to_world[:3, 1] = up
            camera_to_world[:3, 2] = -forward
            camera_to_world[:3, 3] = cam_pos
            
            camera = Camera(
                image_width=self.image_width,
                image_height=self.image_height,
                fx=self.focal_length,
                fy=self.focal_length,
                cx=self.image_width / 2.0,
                cy=self.image_height / 2.0,
                world_to_camera=torch.inverse(camera_to_world),
                camera_to_world=camera_to_world
            )
            
            # Generate synthetic image (checkerboard pattern)
            image = self._generate_synthetic_image(camera)
            
            cameras.append(camera)
            images.append(image)
        
        return cameras, images
    
    def _generate_synthetic_image(self, camera: Camera) -> torch.Tensor:
        """Generate synthetic checkerboard image"""
        # Create checkerboard pattern
        checker_size = 50
        pattern = np.indices((self.image_height, self.image_width))
        pattern = (pattern[0] // checker_size + pattern[1] // checker_size) % 2
        
        # Create RGB image
        image = np.stack([pattern, pattern, pattern], axis=0).astype(np.float32)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0.0, 1.0)
        
        return torch.from_numpy(image).float()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'image': self.images[idx],
            'camera': self.cameras[idx],
            'idx': idx,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'object_center': self.object_center
        }
    
    def get_closeup_test_samples(self):
        """Get close-up test samples for compatibility"""
        # Create some close-up samples from existing data
        closeup_samples = []
        for i in range(min(5, len(self.cameras))):
            closeup_samples.append({
                'image': self.images[i],
                'camera': self.cameras[i],
                'idx': i,
                'object_center': self.object_center
            })
        return closeup_samples
    
    def warp_image(self, source_image: torch.Tensor, source_camera, target_camera) -> torch.Tensor:
        """
        Simple warp function for synthetic data
        For real implementation, use the one in CloseUpDataset
        """
        # For synthetic data, just return the source image (simplified)
        # In practice, this would involve 3D reprojection
        return source_image.clone()


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the dataset
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader object
    """
    def collate_fn(batch):
        """Custom collate function for camera objects"""
        images = torch.stack([item['image'] for item in batch])
        cameras = [item['camera'] for item in batch]
        indices = [item['idx'] for item in batch]
        
        return {
            'images': images,
            'cameras': cameras,
            'indices': indices,
            'image_width': batch[0]['image_width'],
            'image_height': batch[0]['image_height']
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == '__main__':
    # Test dataset functionality
    
    # Test synthetic dataset
    print("Testing synthetic dataset...")
    synthetic_dataset = SyntheticDataset(num_samples=10)
    
    sample = synthetic_dataset[0]
    print(f"Synthetic sample image shape: {sample['image'].shape}")
    print(f"Synthetic sample camera position: {sample['camera'].camera_center}")
    
    # Test dataloader
    dataloader = create_dataloader(synthetic_dataset, batch_size=2)
    batch = next(iter(dataloader))
    
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch cameras count: {len(batch['cameras'])}")
    
    print("Dataset test completed!")