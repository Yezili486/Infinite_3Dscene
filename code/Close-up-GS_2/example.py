#!/usr/bin/env python3
"""
Example usage of Close-up-GS implementation
Demonstrates different scenarios and use cases
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our implementation
from main import (
    ProgressiveGaussianSplatting, 
    EvaluationMetrics,
    CameraParams, 
    ViewInfo,
    load_llff_dataset
)

def example_basic_usage():
    """Basic usage example with synthetic data"""
    print("=== Basic Usage Example ===")
    
    # Initialize Close-up-GS
    closeup_gs = ProgressiveGaussianSplatting()
    
    # Create synthetic training views
    training_views = create_synthetic_scene()
    
    # Set object center
    object_center = torch.tensor([0., 0., 0.], device=closeup_gs.gaussians.device if closeup_gs.gaussians else torch.device('cpu'), dtype=torch.float32)
    closeup_gs.view_selector.object_center = object_center
    
    # Train baseline (reduced iterations for demo)
    print("Training baseline 3DGS...")
    closeup_gs.train_baseline(training_views, iterations=500)
    
    # Run one progressive round
    print("Running progressive expansion...")
    all_rounds = closeup_gs.progressive_expansion(training_views, rounds=1)
    
    # Evaluate results
    evaluator = EvaluationMetrics()
    results = evaluator.evaluate_progressive_results(all_rounds)
    
    print("Results:", results)
    print("Basic example completed!")

def example_with_different_scales():
    """Example showing different close-up scales"""
    print("\n=== Different Scales Example ===")
    
    closeup_gs = ProgressiveGaussianSplatting()
    training_views = create_synthetic_scene()
    
    # Set custom distance factors for different scales
    closeup_gs.distance_factors = [1/2, 1/4, 1/8]  # 2x, 4x, 8x closer
    
    object_center = torch.tensor([0., 0., 0.], device=torch.device('cpu'), dtype=torch.float32)
    closeup_gs.view_selector.object_center = object_center
    
    # Train and expand
    closeup_gs.train_baseline(training_views, iterations=300)
    all_rounds = closeup_gs.progressive_expansion(training_views, rounds=2)
    
    print(f"Processed {len(all_rounds)} rounds with custom scales")

def example_evaluation_metrics():
    """Example of comprehensive evaluation"""
    print("\n=== Evaluation Metrics Example ===")
    
    evaluator = EvaluationMetrics()
    
    # Create test images
    pred_image = torch.rand(256, 256, 3)
    target_image = pred_image + 0.1 * torch.randn_like(pred_image)
    reference_images = [torch.rand(256, 256, 3) for _ in range(3)]
    
    # Evaluate single view
    metrics = evaluator.evaluate_view(pred_image, target_image, reference_images)
    
    print("Single view metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def example_custom_view_selection():
    """Example of custom view selection parameters"""
    print("\n=== Custom View Selection Example ===")
    
    closeup_gs = ProgressiveGaussianSplatting()
    
    # Customize view selection parameters
    closeup_gs.n_anchor_views = 6      # Fewer anchor views
    closeup_gs.n_frontier_views = 12   # Fewer frontier views
    closeup_gs.n_sampled_views = 50    # Fewer sampled views
    closeup_gs.n_update_views = 6      # Fewer views to update
    
    training_views = create_synthetic_scene(n_views=15)
    object_center = torch.tensor([0., 0., 0.], device=torch.device('cpu'), dtype=torch.float32)
    closeup_gs.view_selector.object_center = object_center
    
    # Test view selection
    frontier_views = closeup_gs.view_selector.place_frontier_views(
        training_views[:6], distance_factor=0.5
    )
    
    anchor_indices = closeup_gs.view_selector.select_anchor_views(
        training_views, frontier_views, k=6
    )
    
    print(f"Selected {len(anchor_indices)} anchor views from {len(training_views)} training views")
    print(f"Created {len(frontier_views)} frontier views")

def create_synthetic_scene(n_views=10, radius=2.0):
    """Create a synthetic scene with multiple views"""
    device = torch.device('cpu')  # Use CPU for example
    
    # Camera intrinsics
    focal = 400.0
    cx, cy = 200.0, 150.0
    W, H = 400, 300
    
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    
    views = []
    object_center = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)
    
    for i in range(n_views):
        # Circular camera positions
        angle = 2 * np.pi * i / n_views
        height = 0.3 * np.sin(angle * 2)
        
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
        
        # Camera matrices
        R = torch.stack([right, up, -forward], dim=0)
        T = -R @ cam_pos
        
        camera = CameraParams(R=R, T=T, K=K, width=W, height=H)
        
        # Synthetic image (colorful pattern)
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create interesting synthetic pattern
        pattern = torch.stack([
            0.5 + 0.3 * torch.sin(10 * X + angle),
            0.5 + 0.3 * torch.cos(10 * Y + angle),
            0.5 + 0.3 * torch.sin(5 * (X + Y) + angle)
        ], dim=2).permute(1, 0, 2)  # (H, W, 3)
        
        # Synthetic depth (sphere-like)
        center_x, center_y = W//2, H//2
        xx = torch.arange(W, device=device) - center_x
        yy = torch.arange(H, device=device) - center_y
        XX, YY = torch.meshgrid(xx, yy, indexing='ij')
        depth = radius * 0.8 + 0.2 * torch.exp(-(XX**2 + YY**2) / (100**2))
        depth = depth.T  # (H, W)
        
        view = ViewInfo(camera=camera, image=pattern, depth=depth)
        views.append(view)
    
    return views

def example_save_and_visualize():
    """Example of saving and visualizing results"""
    print("\n=== Save and Visualize Example ===")
    
    # Create output directory
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic scene
    training_views = create_synthetic_scene(n_views=8)
    
    # Save training views
    training_dir = output_dir / "training_views"
    training_dir.mkdir(exist_ok=True)
    
    for i, view in enumerate(training_views[:3]):
        img = view.image.detach().cpu().numpy()
        depth = view.depth.detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.imshow(img)
        ax1.set_title(f"Training View {i} - Image")
        ax1.axis('off')
        
        ax2.imshow(depth, cmap='viridis')
        ax2.set_title(f"Training View {i} - Depth")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(training_dir / f"view_{i}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved training views to {training_dir}")
    
    # Demonstrate close-up artifacts
    closeup_gs = ProgressiveGaussianSplatting()
    closeup_gs.view_selector.object_center = torch.tensor([0., 0., 0.], device=torch.device('cpu'))
    
    # Quick baseline training
    closeup_gs.train_baseline(training_views, iterations=200)
    
    # Create close-up view (2x closer)
    test_camera = training_views[0].camera
    object_center = torch.tensor([0., 0., 0.], device=torch.device('cpu'), dtype=torch.float32)
    close_pos = object_center + (object_center - (-test_camera.R.T @ test_camera.T)) / 2
    
    # Look at matrix
    forward = object_center - close_pos
    forward = forward / torch.norm(forward)
    up = torch.tensor([0., 1., 0.], device=torch.device('cpu'), dtype=torch.float32)
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    close_R = torch.stack([right, up, -forward], dim=0)
    close_T = -close_R @ close_pos
    
    close_camera = CameraParams(
        R=close_R, T=close_T, K=test_camera.K,
        width=test_camera.width, height=test_camera.height
    )
    
    # Render close-up view
    baseline_closeup = closeup_gs.renderer.render(closeup_gs.gaussians, close_camera)
    
    # Save comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(training_views[0].image.detach().cpu().numpy())
    ax1.set_title("Training View (Distant)")
    ax1.axis('off')
    
    ax2.imshow(baseline_closeup['image'].detach().cpu().numpy())
    ax2.set_title("Baseline Close-up (2x closer)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "closeup_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved close-up comparison to {output_dir}/closeup_comparison.png")

def main():
    """Run all examples"""
    print("Close-up-GS Examples\n")
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Basic usage example failed: {e}")
    
    try:
        example_with_different_scales()
    except Exception as e:
        print(f"Different scales example failed: {e}")
    
    try:
        example_evaluation_metrics()
    except Exception as e:
        print(f"Evaluation metrics example failed: {e}")
    
    try:
        example_custom_view_selection()
    except Exception as e:
        print(f"Custom view selection example failed: {e}")
    
    try:
        example_save_and_visualize()
    except Exception as e:
        print(f"Save and visualize example failed: {e}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 