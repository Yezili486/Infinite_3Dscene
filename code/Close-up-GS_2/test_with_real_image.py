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

def create_camera_params(width=400, height=300, fov=60):
    """Create camera parameters for rendering"""
    
    # Intrinsic matrix
    focal_length = width / (2 * np.tan(np.radians(fov / 2)))
    K = torch.tensor([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Extrinsic matrix (looking at origin from different positions)
    R = torch.eye(3, dtype=torch.float32)
    T = torch.tensor([0, 0, -3], dtype=torch.float32)  # 3 units away from origin
    
    return CameraParams(R=R, T=T, K=K, width=width, height=height)

def create_multiple_views_from_single_image(image_path: str, n_views=8):
    """Create multiple views from a single image by simulating different camera positions"""
    
    views = []
    
    for i in range(n_views):
        # Create camera parameters with different positions
        angle = 2 * np.pi * i / n_views
        radius = 3.0
        
        # Position camera in a circle around the object
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = -2.0
        
        # Create rotation matrix to look at origin
        forward = torch.tensor([-x, -y, -z], dtype=torch.float32)
        forward = forward / torch.norm(forward)
        
        right = torch.tensor([1, 0, 0], dtype=torch.float32)
        up = torch.cross(forward, right)
        up = up / torch.norm(up)
        right = torch.cross(up, forward)
        
        R = torch.stack([right, up, -forward], dim=1)
        T = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Create camera parameters
        camera_params = create_camera_params()
        camera_params.R = R
        camera_params.T = T
        
        # Load image as view
        view = load_real_image_as_view(image_path, camera_params)
        views.append(view)
    
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
    
    # Initialize Close-up-GS
    print("Initializing Close-up-GS...")
    closeup_gs = ProgressiveGaussianSplatting()
    
    # Create multiple views from the single image
    print("Creating multiple views from single image...")
    training_views = create_multiple_views_from_single_image(image_path, n_views=6)
    
    print(f"Created {len(training_views)} training views")
    
    # Set object center (center of the image)
    object_center = torch.tensor([0., 0., 0.], device=device, dtype=torch.float32)
    closeup_gs.view_selector.object_center = object_center
    
    # Train baseline (reduced iterations for faster testing)
    print("Training baseline 3DGS...")
    print("This may take a while - you'll see progress updates every 10 iterations...")
    closeup_gs.train_baseline(training_views, iterations=10)
    
    # Run progressive expansion
    print("Running progressive expansion...")
    all_rounds = closeup_gs.progressive_expansion(training_views, rounds=2)
    
    # Evaluate results
    print("Evaluating results...")
    evaluator = EvaluationMetrics()
    results = evaluator.evaluate_progressive_results(all_rounds)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path("real_image_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save evaluation results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save some rendered views
    print("Saving rendered views...")
    for round_idx, round_views in enumerate(all_rounds):
        round_dir = output_dir / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)
        
        for view_idx, view in enumerate(round_views[:3]):  # Save first 3 views
            if view.image is not None:
                # Convert to numpy and save
                image_np = (view.image.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(round_dir / f"view_{view_idx}.png"), 
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
            print(f"{metric}: {value:.4f}")
    
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