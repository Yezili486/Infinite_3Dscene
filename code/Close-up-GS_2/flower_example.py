#!/usr/bin/env python3
"""
Flower Scene Example for Close-up-GS
Demonstrates progressive close-up synthesis on a realistic flower scene
Similar to the examples shown in the Close-up-GS paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy.ndimage import gaussian_filter

# Import our implementation
from main import (
    ProgressiveGaussianSplatting, 
    EvaluationMetrics,
    CameraParams, 
    ViewInfo,
    GaussianRenderer
)

def create_realistic_flower_scene(n_views=16, image_size=(200, 150)):
    """Create a realistic flower scene with detailed geometry and textures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W, H = image_size
    
    print(f"Creating realistic flower scene with {n_views} views...")
    
    # Camera intrinsics
    focal = 500.0
    cx, cy = W//2, H//2
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], 
                     device=device, dtype=torch.float32)
    
    # Flower center (object of interest)
    flower_center = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)
    
    # Create circular camera trajectory around flower (distant views)
    radius = 2.5  # Distance from flower
    views = []
    
    for i in range(n_views):
        # Circular trajectory with some height variation
        angle = 2 * np.pi * i / n_views
        height = 0.3 * np.sin(angle * 2) + 0.2  # Varying height
        
        # Camera position
        cam_pos = torch.tensor([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ], device=device, dtype=torch.float32)
        
        # Look at flower center
        forward = flower_center - cam_pos
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Camera matrices
        R = torch.stack([right, up, -forward], dim=0)
        T = -R @ cam_pos
        
        camera = CameraParams(R=R, T=T, K=K, width=W, height=H)
        
        # Generate realistic flower image and depth
        image, depth = generate_flower_image_and_depth(camera, flower_center, W, H, device)
        
        view = ViewInfo(camera=camera, image=image, depth=depth)
        views.append(view)
    
    print(f"Created {len(views)} training views of flower scene")
    return views, flower_center

def generate_flower_image_and_depth(camera, flower_center, W, H, device):
    """Generate a realistic flower image with proper depth"""
    
    # Create coordinate grids
    x = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
    y = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X, Y = X.T, Y.T  # (H, W)
    
    # Distance from center
    center_x, center_y = 0.0, -0.1  # Flower slightly below center
    dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create flower structure
    flower_radius = 0.4
    petal_count = 5
    
    # Flower petals (sinusoidal pattern)
    angle = torch.atan2(Y - center_y, X - center_x)
    petal_pattern = 0.8 + 0.3 * torch.sin(petal_count * angle)
    flower_mask = dist_from_center < (flower_radius * petal_pattern)
    
    # Flower center
    center_mask = dist_from_center < 0.08
    
    # Leaves and background foliage
    leaf_pattern1 = ((X + 0.6)**2 + (Y + 0.4)**2) < 0.3
    leaf_pattern2 = ((X - 0.5)**2 + (Y + 0.3)**2) < 0.25
    leaf_pattern3 = ((X + 0.3)**2 + (Y - 0.6)**2) < 0.2
    foliage_mask = leaf_pattern1 | leaf_pattern2 | leaf_pattern3
    
    # Background
    background_mask = ~(flower_mask | foliage_mask)
    
    # Initialize image
    image = torch.zeros(H, W, 3, device=device, dtype=torch.float32)
    
    # Flower colors (red/pink flower like in the image)
    flower_base_color = torch.tensor([0.9, 0.2, 0.3], device=device, dtype=torch.float32)  # Red
    flower_highlight = torch.tensor([1.0, 0.6, 0.7], device=device, dtype=torch.float32)  # Pink highlight
    flower_center_color = torch.tensor([1.0, 0.9, 0.4], device=device, dtype=torch.float32)  # Yellow center
    
    # Add gradient and texture to flower
    flower_gradient = 1.0 - (dist_from_center / flower_radius) * 0.5
    flower_gradient = torch.clamp(flower_gradient, 0.3, 1.0)
    
    # Petal texture with some noise
    petal_texture = 0.8 + 0.2 * torch.sin(8 * angle) * torch.cos(12 * dist_from_center)
    petal_texture = torch.clamp(petal_texture, 0.5, 1.0)
    
    # Apply flower colors
    flower_color = flower_base_color.unsqueeze(0).unsqueeze(0) * flower_gradient.unsqueeze(2) * petal_texture.unsqueeze(2)
    image[flower_mask] = flower_color[flower_mask]
    
    # Flower center (stamens)
    image[center_mask] = flower_center_color
    
    # Foliage colors (green leaves)
    leaf_base_color = torch.tensor([0.2, 0.6, 0.3], device=device, dtype=torch.float32)
    leaf_dark_color = torch.tensor([0.1, 0.4, 0.2], device=device, dtype=torch.float32)
    
    # Leaf gradient and texture
    leaf_texture = 0.7 + 0.3 * torch.sin(15 * X) * torch.cos(15 * Y)
    leaf_texture = torch.clamp(leaf_texture, 0.4, 1.0)
    
    # Apply leaf colors
    leaf_color = leaf_base_color.unsqueeze(0).unsqueeze(0) * leaf_texture.unsqueeze(2)
    image[foliage_mask] = leaf_color[foliage_mask]
    
    # Background (darker green, blurred)
    bg_color = torch.tensor([0.15, 0.3, 0.2], device=device, dtype=torch.float32)
    bg_texture = 0.5 + 0.5 * torch.sin(8 * X + 3) * torch.cos(8 * Y + 2)
    bg_texture = torch.clamp(bg_texture, 0.3, 0.8)
    
    bg_final = bg_color.unsqueeze(0).unsqueeze(0) * bg_texture.unsqueeze(2)
    image[background_mask] = bg_final[background_mask]
    
    # Add some overall lighting variation
    lighting = 0.7 + 0.3 * torch.exp(-((X + 0.2)**2 + (Y + 0.1)**2) / 0.8)
    image = image * lighting.unsqueeze(2)
    
    # Create depth map
    depth = torch.ones(H, W, device=device, dtype=torch.float32) * 2.5  # Background depth
    
    # Flower depth (closer)
    flower_depth = 2.0 - 0.3 * flower_gradient
    depth[flower_mask] = flower_depth[flower_mask]
    
    # Flower center depth (slightly raised)
    depth[center_mask] = 1.9
    
    # Foliage depth (intermediate)
    foliage_depth = 2.2 + 0.1 * torch.sin(10 * X) * torch.cos(10 * Y)
    depth[foliage_mask] = foliage_depth[foliage_mask]
    
    # Smooth depth transitions
    depth_np = depth.detach().cpu().numpy()
    depth_smooth = gaussian_filter(depth_np, sigma=1.0)
    depth = torch.tensor(depth_smooth, device=device, dtype=torch.float32)
    
    # Clamp image values
    image = torch.clamp(image, 0.0, 1.0)
    
    return image, depth

def demonstrate_close_up_progression(training_views, flower_center, output_dir):
    """Demonstrate the progressive close-up synthesis on flower scene"""
    
    print("\n=== Demonstrating Close-up Progression on Flower Scene ===")
    
    # Initialize Close-up-GS
    closeup_gs = ProgressiveGaussianSplatting()
    closeup_gs.view_selector.object_center = flower_center
    
    # Custom parameters for flower scene
    closeup_gs.distance_factors = [1/2, 1/4, 1/8]  # 2x, 4x, 8x closer
    closeup_gs.n_anchor_views = 6
    closeup_gs.n_update_views = 4
    
    # Train baseline 3DGS
    print("Training baseline 3DGS...")
    closeup_gs.train_baseline(training_views, iterations=300)
    
    # Save baseline results
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    
    # Create test close-up views at different scales
    test_cameras = create_close_up_test_cameras(training_views[0].camera, flower_center)
    baseline_results = []
    
    print("Rendering baseline close-up views...")
    for i, (scale, test_camera) in enumerate(test_cameras):
        rendered = closeup_gs.renderer.render(closeup_gs.gaussians, test_camera)
        baseline_results.append((scale, rendered['image']))
        
        # Save baseline result
        img_np = rendered['image'].detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(img_np)
        plt.title(f"Baseline Close-up ({scale})")
        plt.axis('off')
        plt.savefig(baseline_dir / f"baseline_{scale}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Run progressive expansion
    print("Running progressive expansion...")
    all_rounds_views = closeup_gs.progressive_expansion(training_views, rounds=2)
    
    # Save progressive results
    for round_idx, round_views in enumerate(all_rounds_views):
        round_dir = output_dir / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)
        
        # Save sample views from this round
        for view_idx, view in enumerate(round_views[:3]):
            if view.image is not None:
                img_np = view.image.detach().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(img_np)
                plt.title(f"Round {round_idx}, View {view_idx}")
                plt.axis('off')
                plt.savefig(round_dir / f"view_{view_idx}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Render final close-up views after progressive enhancement
    print("Rendering final enhanced close-up views...")
    final_results = []
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    
    for i, (scale, test_camera) in enumerate(test_cameras):
        rendered = closeup_gs.renderer.render(closeup_gs.gaussians, test_camera)
        final_results.append((scale, rendered['image']))
        
        # Save final result
        img_np = rendered['image'].detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(img_np)
        plt.title(f"Close-up-GS Result ({scale})")
        plt.axis('off')
        plt.savefig(final_dir / f"final_{scale}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create comparison figure
    create_progression_comparison(training_views[0], baseline_results, final_results, output_dir)
    
    return baseline_results, final_results

def create_close_up_test_cameras(reference_camera, flower_center):
    """Create test cameras at different close-up scales"""
    test_cameras = []
    scales = ["2x_closer", "4x_closer", "8x_closer"]
    distance_factors = [1/2, 1/4, 1/8]
    
    # Reference camera position
    ref_pos = -reference_camera.R.T @ reference_camera.T
    
    for scale, factor in zip(scales, distance_factors):
        # Move camera closer to flower
        direction = flower_center - ref_pos
        new_distance = torch.norm(direction) * factor
        new_direction = direction / torch.norm(direction)
        new_pos = flower_center - new_direction * new_distance
        
        # Create look-at matrix
        forward = flower_center - new_pos
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0., 1., 0.], device=flower_center.device, dtype=torch.float32)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        R = torch.stack([right, up, -forward], dim=0)
        T = -R @ new_pos
        
        test_camera = CameraParams(
            R=R, T=T, K=reference_camera.K,
            width=reference_camera.width, height=reference_camera.height
        )
        
        test_cameras.append((scale, test_camera))
    
    return test_cameras

def create_progression_comparison(training_view, baseline_results, final_results, output_dir):
    """Create a comprehensive comparison showing the progression"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Training view
    train_img = training_view.image.detach().cpu().numpy()
    axes[0, 0].imshow(train_img)
    axes[0, 0].set_title("Training View\n(Distant)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Add flower region indicator
    H, W = train_img.shape[:2]
    rect = plt.Rectangle((W//2-30, H//2-20), 60, 40, 
                        linewidth=2, edgecolor='white', facecolor='none')
    axes[0, 0].add_patch(rect)
    
    # Baseline results
    for i, (scale, img) in enumerate(baseline_results):
        if i < 2:  # Show first two scales
            img_np = img.detach().cpu().numpy()
            axes[0, i+1].imshow(img_np)
            axes[0, i+1].set_title(f"Baseline\n{scale}", fontsize=12)
            axes[0, i+1].axis('off')
    
    # Final results  
    for i, (scale, img) in enumerate(final_results):
        if i < 2:  # Show first two scales
            img_np = img.detach().cpu().numpy()
            axes[1, i+1].imshow(img_np)
            axes[1, i+1].set_title(f"Close-up-GS\n{scale}", fontsize=12, fontweight='bold')
            axes[1, i+1].axis('off')
    
    # Difference maps
    for i in range(min(2, len(baseline_results))):
        baseline_img = baseline_results[i][1].detach().cpu().numpy()
        final_img = final_results[i][1].detach().cpu().numpy()
        diff = np.abs(final_img - baseline_img)
        diff_scaled = diff / diff.max() if diff.max() > 0 else diff
        
        axes[2, i+1].imshow(diff_scaled)
        axes[2, i+1].set_title(f"Improvement\n{baseline_results[i][0]}", fontsize=12)
        axes[2, i+1].axis('off')
    
    # Hide unused subplots
    for i in range(3):
        for j in range(3):
            if (i == 1 and j == 0) or (i == 2 and j == 0):
                axes[i, j].axis('off')
    
    # Add text descriptions
    axes[1, 0].text(0.5, 0.7, "Progressive\nRefinement", ha='center', va='center', 
                   transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold')
    axes[1, 0].text(0.5, 0.5, "✓ See3D Integration", ha='center', va='center', 
                   transform=axes[1, 0].transAxes, fontsize=10)
    axes[1, 0].text(0.5, 0.4, "✓ View Expansion", ha='center', va='center', 
                   transform=axes[1, 0].transAxes, fontsize=10)
    axes[1, 0].text(0.5, 0.3, "✓ Densification", ha='center', va='center', 
                   transform=axes[1, 0].transAxes, fontsize=10)
    
    axes[2, 0].text(0.5, 0.5, "Quality\nImprovement", ha='center', va='center', 
                   transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', color='green')
    
    plt.suptitle("Close-up-GS: Progressive Flower Scene Enhancement", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "flower_progression_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()

def evaluate_flower_scene_quality(baseline_results, final_results, output_dir):
    """Evaluate quality improvements in the flower scene"""
    
    print("\n=== Evaluating Flower Scene Quality ===")
    
    evaluator = EvaluationMetrics()
    
    results = {
        'scales': [],
        'baseline_quality': [],
        'final_quality': [],
        'improvement': []
    }
    
    for i, ((scale_b, baseline_img), (scale_f, final_img)) in enumerate(zip(baseline_results, final_results)):
        assert scale_b == scale_f, "Scale mismatch"
        
        # Evaluate no-reference quality (since we don't have ground truth close-ups)
        baseline_quality = evaluator.compute_meta_iqa(baseline_img)
        final_quality = evaluator.compute_meta_iqa(final_img)
        improvement = final_quality - baseline_quality
        
        results['scales'].append(scale_b)
        results['baseline_quality'].append(baseline_quality)
        results['final_quality'].append(final_quality)
        results['improvement'].append(improvement)
        
        print(f"{scale_b:12} | Baseline: {baseline_quality:.4f} | Final: {final_quality:.4f} | Improvement: {improvement:+.4f}")
    
    # Save evaluation results
    import json
    with open(output_dir / "flower_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create quality comparison plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results['scales']))
    width = 0.35
    
    plt.bar(x - width/2, results['baseline_quality'], width, label='Baseline 3DGS', alpha=0.7)
    plt.bar(x + width/2, results['final_quality'], width, label='Close-up-GS', alpha=0.7)
    
    plt.xlabel('Close-up Scale')
    plt.ylabel('Quality Score')
    plt.title('Quality Comparison: Baseline vs Close-up-GS')
    plt.xticks(x, results['scales'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results

def main():
    """Main function to run the flower scene example"""
    
    print("=== Close-up-GS: Realistic Flower Scene Example ===")
    print("Reproducing paper-like results on a detailed flower scene\n")
    
    # Create output directory
    output_dir = Path("./flower_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create realistic flower scene
        training_views, flower_center = create_realistic_flower_scene(n_views=12, image_size=(400, 300))
        
        # Save training views
        training_dir = output_dir / "training_views"
        training_dir.mkdir(exist_ok=True)
        
        print("Saving training views...")
        for i, view in enumerate(training_views[:4]):  # Save first 4 views
            img_np = view.image.detach().cpu().numpy()
            depth_np = view.depth.detach().cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(img_np)
            ax1.set_title(f"Training View {i+1} - Image")
            ax1.axis('off')
            
            ax2.imshow(depth_np, cmap='viridis')
            ax2.set_title(f"Training View {i+1} - Depth")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(training_dir / f"training_view_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Demonstrate progressive close-up synthesis
        baseline_results, final_results = demonstrate_close_up_progression(
            training_views, flower_center, output_dir
        )
        
        # Evaluate quality improvements
        evaluation_results = evaluate_flower_scene_quality(baseline_results, final_results, output_dir)
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"✓ Created realistic flower scene with {len(training_views)} training views")
        print(f"✓ Trained baseline 3DGS and ran progressive enhancement")
        print(f"✓ Generated close-up views at multiple scales: {[r[0] for r in final_results]}")
        print(f"✓ Average quality improvement: {np.mean(evaluation_results['improvement']):.4f}")
        print(f"✓ Results saved to: {output_dir.absolute()}")
        
        print(f"\nKey outputs:")
        print(f"  📸 Training views: {training_dir}/")
        print(f"  🎯 Baseline results: {output_dir}/baseline/")
        print(f"  🚀 Final results: {output_dir}/final/")
        print(f"  📊 Main comparison: {output_dir}/flower_progression_comparison.png")
        print(f"  📈 Quality analysis: {output_dir}/quality_comparison.png")
        
    except Exception as e:
        print(f"Error in flower scene example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 