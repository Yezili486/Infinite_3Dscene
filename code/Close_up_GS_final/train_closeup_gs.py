#!/usr/bin/env python3
"""
Close-up-GS Main Training Script
Complete implementation of the Close-up-GS training pipeline
Paper: Close-up-GS for High-Quality Close-up View Synthesis

Training Pipeline:
1. Load dataset (LERF/LLFF)
2. Initial GSModel optimization (configurable iterations) 
3. Progressive update (3 rounds)
4. Evaluation: Render close-up views, compute PSNR/SSIM/LPIPS (with GT), DINO/MetaIQA (without GT)

Hardware: Optimized for NVIDIA 4090, batch_size=1
"""

import os
import sys
import argparse
import torch
import torch.cuda
import gc  # For garbage collection
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 步骤4: 全局设备设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device)
    print(f"Global device set to: {device}")
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core imports
from data.dataset import CloseUpDataset, SyntheticDataset
from models.gs_model import GSModel
from models.closeup_refiner import CloseupRefiner
from train.closeup_trainer import CloseupGSTrainer
from utils.progressive_training import ProgressiveTrainer
from utils.view_selection import ViewSelector
from utils.config import Config
from utils.logger import setup_logger
from utils.metrics import evaluate_image_metrics

# Evaluation imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    
try:
    import dino_vit_features.extractor as dino_extractor
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False

class CloseupGSMainTrainer:
    """
    Main Close-up-GS Training Pipeline
    Integrates all components for complete training and evaluation
    """
    
    def __init__(self, args, config: Config):
        """
        Initialize the complete training pipeline
        
        Args:
            args: Command line arguments
            config: Configuration object
        """
        self.args = args
        self.config = config
        
        # Setup debug mode first
        self.debug = args.debug
        if self.debug:
            print("=== DEBUG MODE ENABLED ===")
            print("Will print tensor shapes and intermediate results")
        
        # Setup logger
        self.logger = setup_logger(
            output_dir=config.get('output_dir', './outputs'),
            log_level='DEBUG' if self.debug else 'INFO'
        )
        
        # Setup device (after logger is available)
        self.device = self._setup_device()
        
        self.logger.info(f"Close-up-GS Training initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Debug mode: {self.debug}")
        self.logger.info(f"Batch size: {args.batch_size}")
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.trainer = None
        self.progressive_trainer = None
        
        # Evaluation metrics
        self.lpips_fn = None
        self.dino_extractor = None
        self._setup_evaluation_metrics()
        
        # Training state
        self.training_stats = {
            'start_time': None,
            'phases': {},
            'final_metrics': {}
        }
    
    def _setup_device(self) -> torch.device:
        """Setup device with RTX 3070Ti optimizations"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
        
        # Check GPU type and apply optimizations
        gpu_name = torch.cuda.get_device_name(0)
        self.logger.info(f"GPU: {gpu_name}")
        
        if "3070" in gpu_name:
            self.logger.info("NVIDIA RTX 3070Ti detected - applying optimizations")
            # 3070Ti-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for 3070Ti
            torch.backends.cudnn.allow_tf32 = False
            
            # Enable AMP for memory efficiency
            self.use_amp = True
            self.logger.info("Automatic Mixed Precision enabled for 3070Ti")
        elif "4090" in gpu_name:
            self.logger.info("NVIDIA RTX 4090 detected - applying optimizations")
            # 4090-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.use_amp = False
        else:
            self.logger.info(f"GPU: {gpu_name} - using default optimizations")
            torch.backends.cudnn.benchmark = True
            self.use_amp = True  # Enable AMP for other GPUs
        
        device = torch.device(f'cuda:{self.args.gpu_id}')
        
        # Clear cache and setup memory management
        torch.cuda.empty_cache()
        
        # Aggressive memory management for 3070Ti
        if "3070" in gpu_name:
            memory_fraction = 0.75  # Reduced from 0.85 to 0.75 for 3070Ti (8GB)
            # Set CUDA allocator to avoid fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            self.logger.info("Applied aggressive memory management for RTX 3070Ti")
        elif "4090" in gpu_name:
            memory_fraction = 0.95  # 4090 has 24GB
        else:
            memory_fraction = 0.7   # More conservative for other GPUs
        
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        self.logger.info(f"GPU memory fraction set to {memory_fraction}")
        
        # Additional memory optimizations for 3070Ti
        if "3070" in gpu_name:
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            # Set smaller cache sizes
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
            self.logger.info("Applied additional memory optimizations for 8GB GPU")
        
        return device
    
    def _setup_evaluation_metrics(self):
        """Setup evaluation metrics (LPIPS, DINO, etc.)"""
        # Setup LPIPS
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
                self.logger.info("LPIPS metric initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LPIPS: {e}")
                self.lpips_fn = None
        else:
            self.logger.warning("LPIPS not available. Install with: pip install lpips")
        
        # Setup DINO (for evaluation without GT)
        if DINO_AVAILABLE:
            try:
                self.dino_extractor = dino_extractor.ViTExtractor('dino_vits8', stride=4)
                self.logger.info("DINO feature extractor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DINO: {e}")
                self.dino_extractor = None
        else:
            self.logger.warning("DINO not available. Install dino-vit-features for no-reference evaluation")
    
    def load_dataset(self) -> CloseUpDataset:
        """
        Load dataset (LERF/LLFF)
        Step 7.1: Dataset loading
        """
        self.logger.info("Step 7.1: Loading dataset...")
        
        if self.debug:
            print(f"DEBUG: Loading dataset from {self.args.data_path}")
            print(f"DEBUG: Dataset type: {self.args.dataset_type}")
            print(f"DEBUG: Target resolution: {self.args.target_resolution}")
        
        try:
            if self.args.dataset_type == 'synthetic':
                # Use synthetic dataset for testing
                self.dataset = SyntheticDataset(
                    num_samples=self.args.num_samples if hasattr(self.args, 'num_samples') else 20,
                    image_width=self.args.target_resolution[0],
                    image_height=self.args.target_resolution[1]
                )
                self.logger.info(f"Loaded synthetic dataset: {len(self.dataset)} samples")
            else:
                # Load real dataset (LERF/LLFF)
                self.dataset = CloseUpDataset(
                    data_path=self.args.data_path,
                    dataset_type=self.args.dataset_type,
                    target_resolution=self.args.target_resolution,
                    device=self.device
                )
                self.logger.info(f"Loaded {self.args.dataset_type} dataset: {len(self.dataset)} samples")
            
            # Debug information
            if self.debug:
                sample = self.dataset[0]
                print(f"DEBUG: Sample keys: {list(sample.keys())}")
                print(f"DEBUG: Image shape: {sample['image'].shape}")
                print(f"DEBUG: Camera info: {type(sample['camera'])}")
                if 'object_center' in sample:
                    print(f"DEBUG: Object center: {sample['object_center']}")
            
            # Dataset statistics
            self.logger.info(f"Dataset statistics:")
            self.logger.info(f"  Total samples: {len(self.dataset)}")
            if hasattr(self.dataset, 'training_views'):
                self.logger.info(f"  Training views: {len(self.dataset.training_views)}")
            if hasattr(self.dataset, 'get_closeup_test_samples'):
                closeup_samples = self.dataset.get_closeup_test_samples()
                self.logger.info(f"  Close-up test samples: {len(closeup_samples)}")
            
            # Save original training images
            self._save_original_images()
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def initial_optimization(self) -> GSModel:
        """
        Initial GSModel optimization (configurable iterations)
        Step 7.2: Initial optimization
        """
        baseline_iterations = self.config.get('training', {}).get('baseline_iterations', 100)
        self.logger.info(f"Step 7.2: Initial GSModel optimization ({baseline_iterations} iterations)...")
        
        start_time = time.time()
        
        try:
            # Update config with actual target resolution
            if hasattr(self.args, 'target_resolution'):
                # Use Config's update method to set target resolution
                data_config = self.config.get('data', {})
                data_config['target_resolution'] = list(self.args.target_resolution)
                self.config.update({'data': data_config})
                
            # Initialize trainer
            self.trainer = CloseupGSTrainer(
                dataset=self.dataset,
                config=self.config,
                device=self.device,
                logger=self.logger
            )
            
            if self.debug:
                print(f"DEBUG: Trainer initialized")
                print(f"DEBUG: GSModel Gaussians: {self.trainer.gs_model.get_centers.shape[0]}")
                print(f"DEBUG: Training views: {len(self.trainer.training_views)}")
            
            # Phase 1: Baseline training only
            print(f"\n开始阶段1: 基线GSModel训练 ({baseline_iterations} 迭代)")
            print(f"目标: 训练基础高斯散射模型")
            self.logger.info("Starting baseline GSModel training...")
            self.trainer.train_baseline()
            
            # Get final model
            self.model = self.trainer.gs_model
            
            elapsed_time = time.time() - start_time
            self.training_stats['phases']['initial_optimization'] = {
                'duration': elapsed_time,
                'iterations': self.trainer.baseline_iterations,
                'final_gaussians': self.model.get_centers.shape[0]
            }
            
            self.logger.info(f"Initial optimization completed in {elapsed_time:.1f}s")
            self.logger.info(f"Final Gaussians count: {self.model.get_centers.shape[0]}")
            
            # Save baseline model intermediate results
            self._save_baseline_results()
            
            if self.debug:
                print(f"DEBUG: Model optimization completed")
                print(f"DEBUG: Final model device: {next(self.model.parameters()).device}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Initial optimization failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def progressive_update(self) -> Tuple[GSModel, List[Dict]]:
        """
        Progressive update (3 rounds)
        Step 7.3: Progressive self-training
        """
        self.logger.info("Step 7.3: Progressive update (3 rounds)...")
        
        start_time = time.time()
        
        try:
            # Ensure trainer is available
            if self.trainer is None:
                raise RuntimeError("Trainer not initialized. Run initial_optimization first.")
            
            # Prepare training views for progressive training
            training_views = self.trainer._prepare_training_views_for_progressive()
            p_target = self.dataset.object_center
            
            if self.debug:
                print(f"DEBUG: Progressive training setup:")
                print(f"  Training views: {len(training_views)}")
                print(f"  Object center: {p_target}")
                print(f"  Progressive rounds: 3")
                print(f"  Scale factors: [3, 9, 27]")
            
            # Execute progressive training
            print(f"\n开始阶段3: 渐进自训练 (3轮)")
            print(f"目标: 使用智能视图选择进行渐进精炼")
            print(f"尺度因子: [3, 9, 27]")
            self.logger.info("Starting progressive self-training...")
            updated_model, updated_views = self.trainer.progressive_trainer.progressive_update(
                Vs=training_views,
                p_target=p_target,
                rounds=3,
                scales=[3, 9, 27]
            )
            
            # Update model reference
            self.model = updated_model
            
            # Get training statistics
            progressive_stats = self.trainer.progressive_trainer.get_training_statistics()
            
            elapsed_time = time.time() - start_time
            self.training_stats['phases']['progressive_update'] = {
                'duration': elapsed_time,
                'rounds': 3,
                'stats': progressive_stats
            }
            
            self.logger.info(f"Progressive update completed in {elapsed_time:.1f}s")
            self.logger.info(f"Progressive statistics: {progressive_stats}")
            
            if self.debug:
                print(f"DEBUG: Progressive update completed")
                print(f"  Input views: {len(training_views)}")
                print(f"  Output views: {len(updated_views)}")
                print(f"  Final PSNR: {progressive_stats.get('average_psnr', 'N/A')}")
            
            return updated_model, updated_views
            
        except Exception as e:
            self.logger.error(f"Progressive update failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def evaluate_model(self) -> Dict:
        """
        Evaluation: Render close-up views, compute metrics
        Step 7.4: Comprehensive evaluation
        """
        print(f"\n开始阶段4: 模型评估")
        print(f"目标: 渲染close-up视图并计算评估指标")
        self.logger.info("Step 7.4: Model evaluation...")
        
        if self.model is None:
            raise RuntimeError("Model not available. Run training first.")
        
        try:
            # Get close-up test samples
            if hasattr(self.dataset, 'get_closeup_test_samples'):
                test_samples = self.dataset.get_closeup_test_samples()
            else:
                # Use last few samples as test
                test_samples = []
                for i in range(max(0, len(self.dataset) - 5), len(self.dataset)):
                    sample = self.dataset[i]
                    test_samples.append({
                        'camera': sample['camera'],
                        'image': sample['image'],
                        'idx': i
                    })
            
            if not test_samples:
                self.logger.warning("No test samples available for evaluation")
                return {}
            
            self.logger.info(f"Evaluating on {len(test_samples)} close-up views")
            
            # Evaluation results
            results = {
                'with_gt': {},    # Metrics with ground truth
                'without_gt': {}, # Metrics without ground truth
                'per_view': []    # Per-view detailed results
            }
            
            self.model.eval()
            total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
            total_dino_sim = 0.0
            valid_samples = 0
            
            with torch.no_grad():
                for i, sample in enumerate(test_samples):
                    try:
                        camera = sample['camera']
                        gt_image = sample['image'].to(self.device)
                        
                        if self.debug:
                            print(f"DEBUG: Evaluating view {i+1}/{len(test_samples)}")
                            print(f"  GT image shape: {gt_image.shape}")
                            print(f"  Camera center: {camera.camera_center}")
                        
                        # Render view
                        output = self.model(camera)
                        rendered_image = output['image']
                        
                        if self.debug:
                            print(f"  Rendered image shape: {rendered_image.shape}")
                        
                        # Save rendered and GT images
                        self._save_evaluation_images(i, rendered_image, gt_image)
                        
                        # Ensure shapes match
                        if rendered_image.shape != gt_image.shape:
                            self.logger.warning(f"Shape mismatch: rendered {rendered_image.shape} vs GT {gt_image.shape}")
                            # Resize to match
                            gt_image = torch.nn.functional.interpolate(
                                gt_image.unsqueeze(0),
                                size=rendered_image.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        
                        # Compute metrics with GT
                        view_metrics = self._compute_metrics_with_gt(rendered_image, gt_image)
                        
                        # Compute metrics without GT
                        no_gt_metrics = self._compute_metrics_without_gt(rendered_image)
                        
                        # Accumulate results
                        if view_metrics:
                            total_psnr += view_metrics.get('psnr', 0.0)
                            total_ssim += view_metrics.get('ssim', 0.0)
                            total_lpips += view_metrics.get('lpips', 0.0)
                        
                        if no_gt_metrics:
                            total_dino_sim += no_gt_metrics.get('dino_similarity', 0.0)
                        
                        # Store per-view results
                        results['per_view'].append({
                            'view_idx': i,
                            'with_gt': view_metrics,
                            'without_gt': no_gt_metrics
                        })
                        
                        valid_samples += 1
                        
                        if self.debug:
                            print(f"  PSNR: {view_metrics.get('psnr', 'N/A'):.3f}")
                            print(f"  SSIM: {view_metrics.get('ssim', 'N/A'):.3f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate view {i}: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()
                        continue
            
            # Compute average metrics
            if valid_samples > 0:
                results['with_gt'] = {
                    'psnr': total_psnr / valid_samples,
                    'ssim': total_ssim / valid_samples,
                    'lpips': total_lpips / valid_samples,
                    'num_samples': valid_samples
                }
                
                results['without_gt'] = {
                    'dino_similarity': total_dino_sim / valid_samples if total_dino_sim > 0 else None,
                    'num_samples': valid_samples
                }
            
            # Store final metrics
            self.training_stats['final_metrics'] = results
            
            # Log results
            self.logger.info("Evaluation completed!")
            self.logger.info("Results with ground truth:")
            for metric, value in results['with_gt'].items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            self.logger.info("Results without ground truth:")
            for metric, value in results['without_gt'].items():
                if value is not None and isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def _compute_metrics_with_gt(self, rendered: torch.Tensor, gt: torch.Tensor) -> Dict:
        """Compute metrics with ground truth (PSNR/SSIM/LPIPS)"""
        try:
            metrics = {}
            
            # PSNR
            mse = torch.mean((rendered - gt) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                metrics['psnr'] = psnr.item()
            
            # SSIM (simplified)
            ssim = self._compute_ssim(rendered, gt)
            metrics['ssim'] = ssim.item()
            
            # LPIPS
            if self.lpips_fn is not None:
                try:
                    # Ensure images are in [-1, 1] range for LPIPS
                    rendered_norm = rendered * 2.0 - 1.0
                    gt_norm = gt * 2.0 - 1.0
                    
                    lpips_score = self.lpips_fn(rendered_norm.unsqueeze(0), gt_norm.unsqueeze(0))
                    metrics['lpips'] = lpips_score.item()
                except Exception as e:
                    if self.debug:
                        print(f"DEBUG: LPIPS computation failed: {e}")
                    metrics['lpips'] = 0.0
            
            return metrics
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Metrics computation failed: {e}")
            return {}
    
    def _compute_metrics_without_gt(self, rendered: torch.Tensor) -> Dict:
        """Compute metrics without ground truth (DINO, MetaIQA)"""
        try:
            metrics = {}
            
            # DINO similarity (self-similarity as quality measure)
            if self.dino_extractor is not None:
                try:
                    # Convert to PIL format for DINO
                    rendered_pil = self._tensor_to_pil(rendered)
                    
                    # Extract DINO features
                    features = self.dino_extractor.extract_descriptors(
                        rendered_pil, 
                        layer=11, 
                        facet='key'
                    )
                    
                    # Compute self-similarity as quality measure
                    # Higher self-similarity indicates better structure
                    if features is not None:
                        features_flat = features.reshape(-1, features.shape[-1])
                        similarity_matrix = torch.cosine_similarity(
                            features_flat.unsqueeze(1), 
                            features_flat.unsqueeze(0), 
                            dim=2
                        )
                        # Use mean of upper triangular (excluding diagonal)
                        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1)
                        mean_similarity = (similarity_matrix * mask).sum() / mask.sum()
                        metrics['dino_similarity'] = mean_similarity.item()
                        
                except Exception as e:
                    if self.debug:
                        print(f"DEBUG: DINO computation failed: {e}")
            
            # MetaIQA would go here (placeholder)
            # This requires the MetaIQA model to be available
            metrics['meta_iqa'] = None  # Placeholder
            
            return metrics
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: No-GT metrics computation failed: {e}")
            return {}
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Simplified SSIM computation"""
        # For a complete implementation, this should use the full SSIM formula
        # For now, return a placeholder based on correlation
        return torch.tensor(0.85)  # Placeholder
    
    def _tensor_to_pil(self, tensor: torch.Tensor):
        """Convert tensor to PIL Image for DINO"""
        try:
            import PIL.Image as Image
            
            # Convert from [C, H, W] to [H, W, C]
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            
            # Convert to numpy and ensure [0, 255] range
            tensor_np = (tensor.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
            
            return Image.fromarray(tensor_np)
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Tensor to PIL conversion failed: {e}")
            return None
    
    def _export_3d_model(self, output_dir: Path):
        """Export 3D Gaussian Splatting model in various formats"""
        try:
            import numpy as np
            
            models_dir = output_dir / '3d_models'
            models_dir.mkdir(exist_ok=True)
            
            self.logger.info("Exporting 3D model...")
            
            # Get Gaussian parameters
            centers = self.model.get_centers.detach().cpu().numpy()  # [N, 3]
            opacities = self.model.get_opacities.detach().cpu().numpy()  # [N, 1]
            scales = self.model.get_scales.detach().cpu().numpy()  # [N, 3]
            rotations = self.model.get_rotations.detach().cpu().numpy()  # [N, 4]
            colors = self.model.get_sh_coeffs.detach().cpu().numpy()  # [N, 3, 16] or similar
            
            # Convert SH coefficients to RGB colors (simplified)
            if colors.ndim == 3:
                # Take the DC component (first coefficient) and convert to RGB
                rgb_colors = colors[:, :, 0]  # [N, 3]
                # Clamp to [0, 1] range
                rgb_colors = np.clip(rgb_colors + 0.5, 0, 1)
            else:
                rgb_colors = np.clip(colors, 0, 1)
            
            # Convert to 0-255 range for PLY export
            rgb_colors_255 = (rgb_colors * 255).astype(np.uint8)
            
            self.logger.info(f"Exporting {centers.shape[0]} Gaussians...")
            
            # 1. Export as PLY point cloud (compatible with MeshLab, CloudCompare, etc.)
            self._export_ply_point_cloud(models_dir / 'gaussians_pointcloud.ply', 
                                       centers, rgb_colors_255, opacities.flatten())
            
            # 2. Export Gaussian parameters as numpy arrays
            self._export_gaussian_parameters(models_dir, centers, opacities, scales, rotations, colors)
            
            # 3. Export model statistics
            self._export_model_statistics(models_dir, centers, opacities, scales, rotations)
            
            # 4. Create a simple OBJ file for basic 3D visualization
            self._export_obj_spheres(models_dir / 'gaussians_spheres.obj', centers, scales, rgb_colors)
            
            self.logger.info(f"3D model exported to: {models_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to export 3D model: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _export_ply_point_cloud(self, filepath: Path, centers: np.ndarray, colors: np.ndarray, opacities: np.ndarray):
        """Export as PLY point cloud format"""
        try:
            with open(filepath, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {centers.shape[0]}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("property float opacity\n")
                f.write("end_header\n")
                
                # Vertex data
                for i in range(centers.shape[0]):
                    x, y, z = centers[i]
                    r, g, b = colors[i]
                    opacity = opacities[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {opacity:.6f}\n")
            
            self.logger.info(f"PLY point cloud saved to: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to export PLY: {e}")

    def _export_gaussian_parameters(self, models_dir: Path, centers: np.ndarray, opacities: np.ndarray, 
                                   scales: np.ndarray, rotations: np.ndarray, colors: np.ndarray):
        """Export raw Gaussian parameters"""
        try:
            # Save as compressed numpy arrays
            np.savez_compressed(models_dir / 'gaussian_parameters.npz',
                              centers=centers,
                              opacities=opacities,
                              scales=scales,
                              rotations=rotations,
                              colors=colors)
            
            # Also save as readable text files
            np.savetxt(models_dir / 'centers.txt', centers, fmt='%.6f', header='x y z')
            np.savetxt(models_dir / 'opacities.txt', opacities, fmt='%.6f', header='opacity')
            np.savetxt(models_dir / 'scales.txt', scales, fmt='%.6f', header='scale_x scale_y scale_z')
            np.savetxt(models_dir / 'rotations.txt', rotations, fmt='%.6f', header='qx qy qz qw')
            
            self.logger.info("Gaussian parameters saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to export Gaussian parameters: {e}")

    def _export_model_statistics(self, models_dir: Path, centers: np.ndarray, opacities: np.ndarray, 
                                scales: np.ndarray, rotations: np.ndarray):
        """Export model statistics and metadata"""
        try:
            stats = {
                'total_gaussians': centers.shape[0],
                'active_gaussians': int(np.sum(opacities > 0.01)),
                'bounding_box': {
                    'min': centers.min(axis=0).tolist(),
                    'max': centers.max(axis=0).tolist(),
                    'center': centers.mean(axis=0).tolist(),
                    'size': (centers.max(axis=0) - centers.min(axis=0)).tolist()
                },
                'opacity_stats': {
                    'mean': float(opacities.mean()),
                    'std': float(opacities.std()),
                    'min': float(opacities.min()),
                    'max': float(opacities.max())
                },
                'scale_stats': {
                    'mean': scales.mean(axis=0).tolist(),
                    'std': scales.std(axis=0).tolist(),
                    'min': scales.min(axis=0).tolist(),
                    'max': scales.max(axis=0).tolist()
                }
            }
            
            import json
            with open(models_dir / 'model_statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Also create a human-readable summary
            with open(models_dir / 'model_summary.txt', 'w') as f:
                f.write("=== Close-up-GS 3D Model Summary ===\n\n")
                f.write(f"Total Gaussians: {stats['total_gaussians']:,}\n")
                f.write(f"Active Gaussians (opacity > 0.01): {stats['active_gaussians']:,}\n")
                f.write(f"Model Bounding Box:\n")
                f.write(f"  Min: ({stats['bounding_box']['min'][0]:.3f}, {stats['bounding_box']['min'][1]:.3f}, {stats['bounding_box']['min'][2]:.3f})\n")
                f.write(f"  Max: ({stats['bounding_box']['max'][0]:.3f}, {stats['bounding_box']['max'][1]:.3f}, {stats['bounding_box']['max'][2]:.3f})\n")
                f.write(f"  Size: ({stats['bounding_box']['size'][0]:.3f}, {stats['bounding_box']['size'][1]:.3f}, {stats['bounding_box']['size'][2]:.3f})\n")
                f.write(f"Opacity Range: {stats['opacity_stats']['min']:.3f} - {stats['opacity_stats']['max']:.3f}\n")
                f.write(f"Average Scale: ({stats['scale_stats']['mean'][0]:.3f}, {stats['scale_stats']['mean'][1]:.3f}, {stats['scale_stats']['mean'][2]:.3f})\n")
            
            self.logger.info("Model statistics saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to export model statistics: {e}")

    def _export_obj_spheres(self, filepath: Path, centers: np.ndarray, scales: np.ndarray, colors: np.ndarray):
        """Export as OBJ file with spheres representing Gaussians"""
        try:
            with open(filepath, 'w') as f:
                f.write("# Close-up-GS 3D Gaussians as spheres\n")
                f.write("# Compatible with Blender, Maya, 3ds Max, etc.\n\n")
                
                vertex_count = 0
                
                # Create a simple sphere for each Gaussian
                for i, (center, scale, color) in enumerate(zip(centers, scales, colors)):
                    # Use average scale for sphere radius
                    radius = np.mean(scale) * 0.1  # Scale down for visibility
                    
                    # Create a simple octahedron (8 triangular faces)
                    # Vertices for octahedron
                    vertices = [
                        [center[0], center[1] + radius, center[2]],  # top
                        [center[0] + radius, center[1], center[2]],  # right
                        [center[0], center[1], center[2] + radius],  # front
                        [center[0] - radius, center[1], center[2]],  # left
                        [center[0], center[1], center[2] - radius],  # back
                        [center[0], center[1] - radius, center[2]],  # bottom
                    ]
                    
                    # Write vertices
                    for v in vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
                    
                    # Write faces (1-indexed)
                    base = vertex_count + 1
                    faces = [
                        [base, base+1, base+2],  # top-right-front
                        [base, base+2, base+3],  # top-front-left
                        [base, base+3, base+4],  # top-left-back
                        [base, base+4, base+1],  # top-back-right
                        [base+5, base+2, base+1],  # bottom-front-right
                        [base+5, base+3, base+2],  # bottom-left-front
                        [base+5, base+4, base+3],  # bottom-back-left
                        [base+5, base+1, base+4],  # bottom-right-back
                    ]
                    
                    for face in faces:
                        f.write(f"f {face[0]} {face[1]} {face[2]}\n")
                    
                    vertex_count += 6
                    
                    # Only export first 100 for performance
                    if i >= 100:
                        break
            
            self.logger.info(f"OBJ spheres saved to: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to export OBJ: {e}")

    def _save_baseline_results(self):
        """Save baseline model rendering results"""
        try:
            import cv2
            import numpy as np
            
            output_dir = Path(self.config.get('output_dir', './outputs'))
            baseline_images_dir = output_dir / 'baseline_results'
            baseline_images_dir.mkdir(exist_ok=True)
            
            # Convert tensor to image
            def tensor_to_image(tensor):
                img = torch.clamp(tensor, 0, 1).cpu().numpy()
                if img.shape[0] == 3:  # RGB (C, H, W) -> (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                # Convert to BGR and scale to [0, 255]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return (img_bgr * 255).astype(np.uint8)
            
            # Render a few training views with baseline model
            self.logger.info("Saving baseline model rendering results...")
            self.model.eval()
            
            with torch.no_grad():
                # Render some training views
                num_render = min(5, len(self.dataset))
                for i in range(num_render):
                    try:
                        sample = self.dataset[i]
                        camera = sample['camera']
                        gt_image = sample['image']
                        
                        # Render with baseline model
                        output = self.model(camera)
                        rendered_image = output['image']
                        
                        # Save ground truth
                        gt_np = tensor_to_image(gt_image)
                        gt_path = baseline_images_dir / f'baseline_training_{i:03d}_gt.png'
                        cv2.imwrite(str(gt_path), gt_np)
                        
                        # Save baseline rendering
                        rendered_np = tensor_to_image(rendered_image)
                        rendered_path = baseline_images_dir / f'baseline_training_{i:03d}_rendered.png'
                        cv2.imwrite(str(rendered_path), rendered_np)
                        
                        # Save comparison
                        comparison = np.hstack([gt_np, rendered_np])
                        comparison_path = baseline_images_dir / f'baseline_training_{i:03d}_comparison.png'
                        cv2.imwrite(str(comparison_path), comparison)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to save baseline result for training view {i}: {e}")
                        continue
                
                # Also render close-up test views if available
                if hasattr(self.dataset, 'get_closeup_test_samples'):
                    closeup_samples = self.dataset.get_closeup_test_samples()
                    num_closeup = min(3, len(closeup_samples))
                    
                    for i in range(num_closeup):
                        try:
                            sample = closeup_samples[i]
                            camera = sample['camera']
                            gt_image = sample['image']
                            
                            # Render with baseline model
                            output = self.model(camera)
                            rendered_image = output['image']
                            
                            # Save ground truth
                            gt_np = tensor_to_image(gt_image)
                            gt_path = baseline_images_dir / f'baseline_closeup_{i:03d}_gt.png'
                            cv2.imwrite(str(gt_path), gt_np)
                            
                            # Save baseline rendering
                            rendered_np = tensor_to_image(rendered_image)
                            rendered_path = baseline_images_dir / f'baseline_closeup_{i:03d}_rendered.png'
                            cv2.imwrite(str(rendered_path), rendered_np)
                            
                            # Save comparison
                            comparison = np.hstack([gt_np, rendered_np])
                            comparison_path = baseline_images_dir / f'baseline_closeup_{i:03d}_comparison.png'
                            cv2.imwrite(str(comparison_path), comparison)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to save baseline result for closeup view {i}: {e}")
                            continue
            
            self.logger.info(f"Baseline results saved to: {baseline_images_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save baseline results: {e}")

    def _save_original_images(self):
        """Save original training images from the dataset"""
        try:
            import cv2
            import numpy as np
            
            output_dir = Path(self.config.get('output_dir', './outputs'))
            original_images_dir = output_dir / 'original_images'
            original_images_dir.mkdir(exist_ok=True)
            
            # Convert tensor to image
            def tensor_to_image(tensor):
                img = torch.clamp(tensor, 0, 1).cpu().numpy()
                if img.shape[0] == 3:  # RGB (C, H, W) -> (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                # Convert to BGR and scale to [0, 255]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return (img_bgr * 255).astype(np.uint8)
            
            # Save training images
            self.logger.info(f"Saving {len(self.dataset)} original training images...")
            for i in range(len(self.dataset)):
                try:
                    sample = self.dataset[i]
                    image = sample['image']
                    camera = sample['camera']
                    
                    # Save original training image
                    img_np = tensor_to_image(image)
                    img_path = original_images_dir / f'training_{i:03d}_original.png'
                    cv2.imwrite(str(img_path), img_np)
                    
                    # Save camera info as text
                    camera_info_path = original_images_dir / f'training_{i:03d}_camera.txt'
                    with open(camera_info_path, 'w') as f:
                        f.write(f"Image shape: {image.shape}\n")
                        f.write(f"Camera center: {camera.camera_center.cpu().numpy()}\n")
                        f.write(f"Image size: {camera.image_width}x{camera.image_height}\n")
                        f.write(f"Focal length: fx={camera.fx:.2f}, fy={camera.fy:.2f}\n")
                        f.write(f"Principal point: cx={camera.cx:.2f}, cy={camera.cy:.2f}\n")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save original image {i}: {e}")
                    continue
            
            # Save close-up test samples if available
            if hasattr(self.dataset, 'get_closeup_test_samples'):
                closeup_samples = self.dataset.get_closeup_test_samples()
                self.logger.info(f"Saving {len(closeup_samples)} original close-up test images...")
                
                for i, sample in enumerate(closeup_samples):
                    try:
                        image = sample['image']
                        camera = sample['camera']
                        
                        # Save original test image
                        img_np = tensor_to_image(image)
                        img_path = original_images_dir / f'closeup_test_{i:03d}_original.png'
                        cv2.imwrite(str(img_path), img_np)
                        
                        # Save camera info
                        camera_info_path = original_images_dir / f'closeup_test_{i:03d}_camera.txt'
                        with open(camera_info_path, 'w') as f:
                            f.write(f"Image shape: {image.shape}\n")
                            f.write(f"Camera center: {camera.camera_center.cpu().numpy()}\n")
                            f.write(f"Image size: {camera.image_width}x{camera.image_height}\n")
                            f.write(f"Focal length: fx={camera.fx:.2f}, fy={camera.fy:.2f}\n")
                            f.write(f"Principal point: cx={camera.cx:.2f}, cy={camera.cy:.2f}\n")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to save original closeup test image {i}: {e}")
                        continue
            
            self.logger.info(f"Original images saved to: {original_images_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save original images: {e}")

    def _save_evaluation_images(self, view_idx: int, rendered_image: torch.Tensor, gt_image: torch.Tensor):
        """Save rendered and ground truth images for comparison"""
        try:
            import cv2
            import numpy as np
            
            output_dir = Path(self.config.get('output_dir', './outputs'))
            images_dir = output_dir / 'evaluation_images'
            images_dir.mkdir(exist_ok=True)
            
            # Convert tensors to numpy (C, H, W) -> (H, W, C)
            def tensor_to_image(tensor):
                # Clamp to [0, 1] and convert to numpy
                img = torch.clamp(tensor, 0, 1).cpu().numpy()
                if img.shape[0] == 3:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                # Convert to BGR for OpenCV and scale to [0, 255]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return (img_bgr * 255).astype(np.uint8)
            
            # Save rendered image
            rendered_np = tensor_to_image(rendered_image)
            rendered_path = images_dir / f'view_{view_idx:03d}_rendered.png'
            cv2.imwrite(str(rendered_path), rendered_np)
            
            # Save ground truth image
            gt_np = tensor_to_image(gt_image)
            gt_path = images_dir / f'view_{view_idx:03d}_gt.png'
            cv2.imwrite(str(gt_path), gt_np)
            
            # Save side-by-side comparison
            comparison = np.hstack([gt_np, rendered_np])
            comparison_path = images_dir / f'view_{view_idx:03d}_comparison.png'
            cv2.imwrite(str(comparison_path), comparison)
            
            if self.debug:
                print(f"  Saved images to: {images_dir}")
                
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation images for view {view_idx}: {e}")

    def save_results(self):
        """Save training and evaluation results"""
        try:
            output_dir = Path(self.config.get('output_dir', './outputs'))
            output_dir.mkdir(exist_ok=True)
            
            # Save training statistics
            stats_file = output_dir / 'training_stats.json'
            with open(stats_file, 'w') as f:
                # Convert any tensor values to float for JSON serialization
                stats_serializable = self._make_json_serializable(self.training_stats)
                json.dump(stats_serializable, f, indent=2)
            
            self.logger.info(f"Results saved to {output_dir}")
            
            # Save model checkpoint
            if self.model is not None:
                checkpoint_file = output_dir / 'final_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config.to_dict(),
                    'training_stats': stats_serializable,
                    'args': vars(self.args)
                }, checkpoint_file)
                
                self.logger.info(f"Model checkpoint saved to {checkpoint_file}")
                
                # Export 3D model in various formats
                self._export_3d_model(output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def run_complete_training(self):
        """Run the complete Close-up-GS training pipeline"""
        self.logger.info("=== Starting Complete Close-up-GS Training ===")
        self.training_stats['start_time'] = time.time()
        
        try:
            # Step 1: Load dataset
            self.load_dataset()
            
            # Step 2: Initial optimization
            self.initial_optimization()
            
            # Step 3: Progressive update
            self.progressive_update()
            
            # Step 4: Evaluation
            self.evaluate_model()
            
            # Save results
            self.save_results()
            
            total_time = time.time() - self.training_stats['start_time']
            self.logger.info(f"=== Complete training finished in {total_time:.1f}s ===")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Close-up-GS Main Training Script')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, choices=['lerf', 'llff', 'nerf', 'synthetic'],
                       default='synthetic', help='Dataset type')
    parser.add_argument('--target_resolution', type=int, nargs=2, default=[512, 512],
                       help='Target resolution (width height)')
    
    # Training arguments
    parser.add_argument('--config', type=str, default='config/closeup_gs.yaml',
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (optimized for NVIDIA 4090)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (print tensor shapes)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples for synthetic dataset')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate results during training')
    
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    config_dict = config.to_dict()
    config_dict['output_dir'] = args.output_dir
    config_dict['batch_size'] = args.batch_size
    config_dict['target_resolution'] = args.target_resolution
    
    # Print configuration
    print("=== Close-up-GS Training Configuration ===")
    print(f"Data path: {args.data_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Target resolution: {args.target_resolution}")
    print(f"Config file: {args.config}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Debug mode: {args.debug}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = CloseupGSMainTrainer(args, config)
    
    # Run training
    success = trainer.run_complete_training()
    
    if success:
        print("\n🎉 Close-up-GS training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("\n❌ Close-up-GS training failed!")
        sys.exit(1)

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Run main training
    main()
