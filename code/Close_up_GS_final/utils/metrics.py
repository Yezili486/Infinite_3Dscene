"""
Evaluation metrics for Close-up-GS
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
# import lpips  # Optional dependency
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1: torch.Tensor, 
         img2: torch.Tensor, 
         window_size: int = 11,
         size_average: bool = True) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of sliding window
        size_average: Whether to average over spatial dimensions
        
    Returns:
        SSIM value
    """
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create Gaussian window"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0).unsqueeze(0) * g.unsqueeze(0).unsqueeze(1)
    
    # Ensure images are 4D [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    channels = img1.shape[1]
    window = gaussian_window(window_size).repeat(channels, 1, 1, 1)
    window = window.to(img1.device)
    
    # Constants for SSIM calculation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM calculation
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def lpips_metric(img1: torch.Tensor, img2: torch.Tensor, net: str = 'alex') -> torch.Tensor:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity)
    
    Args:
        img1: First image [B, C, H, W] in range [0, 1]
        img2: Second image [B, C, H, W] in range [0, 1]
        net: Network type ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS distance
    """
    if not LPIPS_AVAILABLE:
        # Return dummy value if LPIPS not available
        return torch.tensor(0.5, device=img1.device)
    
    try:
        # Initialize LPIPS model (cached)
        if not hasattr(lpips_metric, 'lpips_model') or lpips_metric.lpips_model is None:
            lpips_metric.lpips_model = lpips.LPIPS(net=net, verbose=False)
            lpips_metric.lpips_model.eval()
            # Move to device immediately after creation
            lpips_metric.lpips_model = lpips_metric.lpips_model.to(img1.device)
        
        # Ensure model is on the correct device
        device = img1.device
        if next(lpips_metric.lpips_model.parameters()).device != device:
            lpips_metric.lpips_model = lpips_metric.lpips_model.to(device)
        
        # Ensure images are in range [-1, 1] for LPIPS
        img1_norm = 2 * img1 - 1
        img2_norm = 2 * img2 - 1
        
        # Add batch dimension if needed
        if img1_norm.dim() == 3:
            img1_norm = img1_norm.unsqueeze(0)
            img2_norm = img2_norm.unsqueeze(0)
        
        with torch.no_grad():
            result = lpips_metric.lpips_model(img1_norm, img2_norm)
            return result.mean()
    
    except Exception as e:
        print(f"LPIPS calculation failed: {e}, returning dummy value")
        return torch.tensor(0.5, device=img1.device)


def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error"""
    return torch.mean((img1 - img2) ** 2)


def mae(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(img1 - img2))


def evaluate_image_metrics(pred: torch.Tensor, 
                          target: torch.Tensor,
                          metrics: list = ['psnr', 'ssim', 'lpips']) -> dict:
    """
    Evaluate multiple image quality metrics
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W]
        target: Target image [B, C, H, W] or [C, H, W]
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    # Ensure images are in [0, 1] range
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    
    if 'mse' in metrics:
        results['mse'] = mse(pred, target).item()
    
    if 'mae' in metrics:
        results['mae'] = mae(pred, target).item()
    
    if 'psnr' in metrics:
        results['psnr'] = psnr(pred, target).item()
    
    if 'ssim' in metrics:
        results['ssim'] = ssim(pred, target).item()
    
    if 'lpips' in metrics:
        try:
            results['lpips'] = lpips_metric(pred, target).item()
        except Exception as e:
            print(f"Warning: LPIPS calculation failed: {e}")
            results['lpips'] = float('nan')
    
    return results


class MetricTracker:
    """Track metrics during training/evaluation"""
    
    def __init__(self, metrics: list = ['psnr', 'ssim', 'lpips']):
        """
        Initialize metric tracker
        
        Args:
            metrics: List of metrics to track
        """
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.values = {metric: [] for metric in self.metrics}
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions
        
        Args:
            pred: Predicted image
            target: Target image
        """
        metric_values = evaluate_image_metrics(pred, target, self.metrics)
        
        for metric in self.metrics:
            if metric in metric_values:
                self.values[metric].append(metric_values[metric])
        
        self.count += 1
    
    def get_averages(self) -> dict:
        """Get average values for all metrics"""
        averages = {}
        for metric in self.metrics:
            if self.values[metric]:
                averages[metric] = np.mean(self.values[metric])
            else:
                averages[metric] = float('nan')
        return averages
    
    def get_latest(self) -> dict:
        """Get latest values for all metrics"""
        latest = {}
        for metric in self.metrics:
            if self.values[metric]:
                latest[metric] = self.values[metric][-1]
            else:
                latest[metric] = float('nan')
        return latest
    
    def __str__(self) -> str:
        """String representation of current averages"""
        averages = self.get_averages()
        return " | ".join([f"{k}: {v:.4f}" for k, v in averages.items()])


if __name__ == '__main__':
    # Test metrics
    print("Testing image quality metrics...")
    
    # Create test images
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + 0.1 * torch.randn_like(img1)  # Add noise
    img2 = torch.clamp(img2, 0.0, 1.0)
    
    # Test individual metrics
    print(f"PSNR: {psnr(img1, img2).item():.2f}")
    print(f"SSIM: {ssim(img1, img2).item():.4f}")
    print(f"MSE: {mse(img1, img2).item():.6f}")
    print(f"MAE: {mae(img1, img2).item():.6f}")
    
    # Test metric evaluation
    results = evaluate_image_metrics(img1, img2, ['psnr', 'ssim', 'mse'])
    print(f"Evaluation results: {results}")
    
    # Test metric tracker
    tracker = MetricTracker(['psnr', 'ssim'])
    tracker.update(img1, img2)
    tracker.update(img1, img2)
    print(f"Tracker averages: {tracker.get_averages()}")
    
    print("Metrics test completed!")
