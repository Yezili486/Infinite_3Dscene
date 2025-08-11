#!/usr/bin/env python3
"""
Test Close-up-GS with real images
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import our implementation
from main import (
    ProgressiveGaussianSplatting, 
    EvaluationMetrics,
    CameraParams, 
    ViewInfo
)

def load_real_image_as_view(image_path: str, camera_params: CameraParams) -> ViewInfo:
    """Load a real image and convert it to a ViewInfo object"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match camera parameters
    image = cv2.resize(image, (camera_params.width, camera_params.height))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).to(device)
    
    # Create depth map (placeholder - you can replace with real depth if available)
    depth = torch.ones((camera_params.height, camera_params.width), 
                      dtype=torch.float32, device=device) * 2.0
    
    # Create mask (all pixels are valid)
    mask = torch.ones((camera_params.height, camera_params.width), 
                     dtype=torch.bool, device=device)
    
    return ViewInfo(
        camera=camera_params,
        image=image_tensor,
        depth=depth,
        mask=mask,
        reliability_score=1.0
    )

def create_camera_params(width=200, height=150, fov=60.0):
    """Create camera parameters with very small resolution to save memory"""
    # Intrinsic matrix - use a reasonable focal length
    # For a 60 degree FOV, focal length should be approximately width/2
    focal_length = width / 2.0  # Simplified focal length calculation
    K = torch.tensor([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Default extrinsic matrix (will be overridden in create_multiple_views)
    R = torch.eye(3, dtype=torch.float32)
    T = torch.tensor([0, 0, 2], dtype=torch.float32)  # Positive Z, camera in front
    
    return CameraParams(R=R, T=T, K=K, width=width, height=height)

def create_distant_training_views(image_path: str, n_views=4):
    """Create DISTANT training views - key insight: train from far away"""
    
    views = []
    
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        
        # DISTANT cameras: 5-6 units away (远距离)
        radius = 5.0 + i * 0.5  # 5.0, 5.5, 6.0, 6.5
        x = radius * np.cos(angle) * 0.2
        y = radius * np.sin(angle) * 0.2  
        z = 3.0 + i * 0.3  # Vary height: 3.0, 3.3, 3.6, 3.9
        
        camera_pos = torch.tensor([x, y, z], dtype=torch.float32)
        target_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
        # Calculate view direction
        forward = target_pos - camera_pos
        forward = forward / torch.norm(forward)
        
        up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(forward, up_world)
        if torch.norm(right) < 1e-6:
            right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        else:
            right = right / torch.norm(right)
        
        up = torch.cross(right, forward)
        up = up / torch.norm(up)
        
        R = torch.stack([right, up, -forward], dim=1)
        T = camera_pos
        
        camera_params = create_camera_params()
        camera_params.R = R
        camera_params.T = T
        
        view = load_real_image_as_view(image_path, camera_params)
        views.append(view)
    
    print(f"Created distant training views at distances: {[5.0 + i * 0.5 for i in range(n_views)]}")
    return views

def create_closeup_target_views(image_path: str, n_views=3):
    """Create CLOSE-UP target views - key insight: test close-up quality"""
    
    views = []
    
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        
        # CLOSE-UP cameras: 1-2 units away (近距离/放大)
        radius = 1.2 + i * 0.3  # 1.2, 1.5, 1.8 - much closer!
        x = radius * np.cos(angle) * 0.4
        y = radius * np.sin(angle) * 0.4
        z = 1.0 + i * 0.2  # 1.0, 1.2, 1.4
        
        camera_pos = torch.tensor([x, y, z], dtype=torch.float32)
        target_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
        # Calculate view direction
        forward = target_pos - camera_pos
        forward = forward / torch.norm(forward)
        
        up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(forward, up_world)
        if torch.norm(right) < 1e-6:
            right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        else:
            right = right / torch.norm(right)
        
        up = torch.cross(right, forward)
        up = up / torch.norm(up)
        
        R = torch.stack([right, up, -forward], dim=1)
        T = camera_pos
        
        camera_params = create_camera_params()
        camera_params.R = R
        camera_params.T = T
        
        view = load_real_image_as_view(image_path, camera_params)
        views.append(view)
    
    print(f"Created close-up target views at distances: {[1.2 + i * 0.3 for i in range(n_views)]}")
    return views

def test_with_real_image():
    """Test Close-up-GS with a real image"""
    
    print("=== Testing Close-up-GS with Real Image ===")
    
    # Check if test image exists
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found!")
        return
    
    print(f"Using test image: {image_path}")
    
    # Initialize Close-up-GS with reduced parameters
    print("Initializing Close-up-GS...")
    closeup_gs = ProgressiveGaussianSplatting(
        scene_bounds=(torch.tensor([-2, -2, -2], dtype=torch.float32), 
                     torch.tensor([2, 2, 2], dtype=torch.float32)),
        sh_degree=3
    )
    
    # Create DISTANT training views (key: train from far away)
    print("Creating DISTANT training views (远距离训练视角)...")
    training_views = create_distant_training_views(image_path, n_views=4)
    
    # Create CLOSE-UP target views (key: test close-up quality)
    print("Creating CLOSE-UP target views (近距离目标视角)...")
    closeup_target_views = create_closeup_target_views(image_path, n_views=3)
    
    print(f"Created {len(training_views)} training views")
    
    # Set object center (center of the image)
    object_center = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)
    closeup_gs.view_selector.object_center = object_center
    
    # Train baseline (increased iterations for better quality)
    print("Training baseline 3DGS...")
    print("This may take a while - you'll see progress updates every 10 iterations...")
    closeup_gs.train_baseline(training_views, iterations=20)  # Increased from 5
    
    # Clear GPU memory after training
    torch.cuda.empty_cache()
    print("Baseline training completed!")
    
    # Run progressive expansion with reduced rounds
    print("Running progressive expansion...")
    all_rounds = closeup_gs.progressive_expansion(training_views, rounds=2)  # Changed back to 2
    
    # 🔥 KEY TEST: Render close-up views and test quality
    print("\n=== 🔍 TESTING CLOSE-UP QUALITY (放大视角质量测试) ===")
    print("Testing if model trained on DISTANT views can render high-quality CLOSE-UP views...")
    
    closeup_rendered_views = []
    for i, target_view in enumerate(closeup_target_views):
        print(f"Rendering close-up view {i+1}/{len(closeup_target_views)} (distance: {1.2 + i * 0.3:.1f} units)...")
        
        # Render the close-up view using the trained model
        rendered = closeup_gs.renderer.render(closeup_gs.gaussians, target_view.camera)
        
        if rendered['image'] is not None:
            closeup_rendered_view = ViewInfo(
                camera=target_view.camera,
                image=rendered['image'],
                depth=rendered.get('depth'),
                reliability_score=1.0
            )
            closeup_rendered_views.append(closeup_rendered_view)
        else:
            print(f"  Warning: Close-up view {i} rendering failed")
    
    # Create comparison results: distant training vs close-up testing
    comparison_rounds = all_rounds + [closeup_rendered_views]  # Add close-up results
    
    # Evaluate results
    print("Evaluating results...")
    evaluator = EvaluationMetrics()
    results = evaluator.evaluate_progressive_results(comparison_rounds)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save results
    output_dir = Path("real_image_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save evaluation results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save some rendered views
    print("Saving rendered views...")
    # Save training rounds
    for round_idx, round_views in enumerate(all_rounds):
        round_dir = output_dir / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)
        
        for view_idx, view in enumerate(round_views[:3]):  # Save first 3 views
            if view.image is not None:
                # Convert to numpy and save
                image_np = (view.image.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(round_dir / f"view_{view_idx}.png"), 
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    # 🔥 Save close-up test results
    if closeup_rendered_views:
        closeup_dir = output_dir / "closeup_test"
        closeup_dir.mkdir(exist_ok=True)
        
        print("Saving CLOSE-UP test results...")
        for view_idx, view in enumerate(closeup_rendered_views):
            if view.image is not None:
                image_np = (view.image.cpu().numpy() * 255).astype(np.uint8)
                distance = 1.2 + view_idx * 0.3
                cv2.imwrite(str(closeup_dir / f"closeup_view_{view_idx}_dist_{distance:.1f}.png"),
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved in: {output_dir.absolute()}")
    print("Real image test completed!")

def visualize_results():
    """Visualize the results"""
    
    output_dir = Path("real_image_test_output")
    if not output_dir.exists():
        print("No results to visualize!")
        return
    
    # Load evaluation results
    results_file = output_dir / "evaluation_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        
        print("\n=== Evaluation Results ===")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    # Show some rendered images
    print("\n=== Rendered Views ===")
    for round_dir in output_dir.glob("round_*"):
        print(f"\n{round_dir.name}:")
        for image_file in round_dir.glob("*.png"):
            print(f"  - {image_file.name}")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run test
    test_with_real_image()
    
    # Visualize results
    visualize_results() 