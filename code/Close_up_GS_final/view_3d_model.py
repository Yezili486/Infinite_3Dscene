#!/usr/bin/env python3
"""
Close-up-GS 3D Model Viewer
Visualize the trained 3D Gaussian Splatting model
"""

import argparse
import numpy as np
from pathlib import Path
import json

def load_model_data(models_dir: Path):
    """Load 3D model data from exported files"""
    print(f"Loading 3D model from: {models_dir}")
    
    # Check available files
    files = {
        'parameters': models_dir / 'gaussian_parameters.npz',
        'statistics': models_dir / 'model_statistics.json',
        'summary': models_dir / 'model_summary.txt',
        'ply': models_dir / 'gaussians_pointcloud.ply',
        'obj': models_dir / 'gaussians_spheres.obj'
    }
    
    print("\n=== Available Files ===")
    for name, path in files.items():
        status = "✓" if path.exists() else "✗"
        print(f"{status} {name}: {path.name}")
    
    data = {}
    
    # Load Gaussian parameters
    if files['parameters'].exists():
        params = np.load(files['parameters'])
        data['centers'] = params['centers']
        data['opacities'] = params['opacities']
        data['scales'] = params['scales']
        data['rotations'] = params['rotations']
        data['colors'] = params['colors']
        print(f"\n✓ Loaded {data['centers'].shape[0]} Gaussians")
    
    # Load statistics
    if files['statistics'].exists():
        with open(files['statistics']) as f:
            data['stats'] = json.load(f)
        print("✓ Loaded model statistics")
    
    return data, files

def print_model_summary(data: dict, files: dict):
    """Print detailed model summary"""
    print("\n" + "="*50)
    print("         CLOSE-UP-GS 3D MODEL SUMMARY")
    print("="*50)
    
    if 'stats' in data:
        stats = data['stats']
        print(f"Total Gaussians: {stats['total_gaussians']:,}")
        print(f"Active Gaussians: {stats['active_gaussians']:,}")
        print(f"Model Bounding Box:")
        print(f"  Min: ({stats['bounding_box']['min'][0]:.3f}, {stats['bounding_box']['min'][1]:.3f}, {stats['bounding_box']['min'][2]:.3f})")
        print(f"  Max: ({stats['bounding_box']['max'][0]:.3f}, {stats['bounding_box']['max'][1]:.3f}, {stats['bounding_box']['max'][2]:.3f})")
        print(f"  Size: ({stats['bounding_box']['size'][0]:.3f}, {stats['bounding_box']['size'][1]:.3f}, {stats['bounding_box']['size'][2]:.3f})")
        print(f"Opacity Range: {stats['opacity_stats']['min']:.3f} - {stats['opacity_stats']['max']:.3f}")
        print(f"Average Scale: ({stats['scale_stats']['mean'][0]:.3f}, {stats['scale_stats']['mean'][1]:.3f}, {stats['scale_stats']['mean'][2]:.3f})")
    
    if 'centers' in data:
        centers = data['centers']
        opacities = data['opacities']
        scales = data['scales']
        
        print(f"\n--- Gaussian Distribution ---")
        print(f"Center range:")
        print(f"  X: {centers[:, 0].min():.3f} to {centers[:, 0].max():.3f}")
        print(f"  Y: {centers[:, 1].min():.3f} to {centers[:, 1].max():.3f}")
        print(f"  Z: {centers[:, 2].min():.3f} to {centers[:, 2].max():.3f}")
        print(f"Opacity distribution:")
        print(f"  Mean: {opacities.mean():.3f} ± {opacities.std():.3f}")
        print(f"  Active (>0.01): {np.sum(opacities > 0.01)}/{len(opacities)}")
        print(f"Scale distribution:")
        print(f"  X: {scales[:, 0].mean():.3f} ± {scales[:, 0].std():.3f}")
        print(f"  Y: {scales[:, 1].mean():.3f} ± {scales[:, 1].std():.3f}")
        print(f"  Z: {scales[:, 2].mean():.3f} ± {scales[:, 2].std():.3f}")

def visualize_with_matplotlib(data: dict):
    """Simple 3D visualization using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if 'centers' not in data:
            print("No Gaussian centers available for visualization")
            return
        
        centers = data['centers']
        opacities = data['opacities'].flatten()
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filter active Gaussians
        active_mask = opacities > 0.01
        active_centers = centers[active_mask]
        active_opacities = opacities[active_mask]
        
        # Sample for performance (max 1000 points)
        if len(active_centers) > 1000:
            indices = np.random.choice(len(active_centers), 1000, replace=False)
            active_centers = active_centers[indices]
            active_opacities = active_opacities[indices]
        
        # Color by opacity
        scatter = ax.scatter(active_centers[:, 0], 
                           active_centers[:, 1], 
                           active_centers[:, 2],
                           c=active_opacities, 
                           cmap='viridis',
                           alpha=0.6,
                           s=20)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Close-up-GS 3D Model\n({len(active_centers)} active Gaussians)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Opacity')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ 3D visualization displayed")
        
    except ImportError:
        print("✗ matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

def export_for_external_tools(data: dict, files: dict, output_dir: Path):
    """Export in formats compatible with external 3D tools"""
    print(f"\n=== Export Information ===")
    
    if files['ply'].exists():
        print(f"✓ PLY Point Cloud: {files['ply']}")
        print("  - Compatible with: MeshLab, CloudCompare, Blender")
        print("  - Open in MeshLab: File → Import Mesh → Select .ply file")
    
    if files['obj'].exists():
        print(f"✓ OBJ Spheres: {files['obj']}")
        print("  - Compatible with: Blender, Maya, 3ds Max, etc.")
        print("  - Open in Blender: File → Import → Wavefront (.obj)")
    
    # Create a quick visualization script
    vis_script = output_dir / 'quick_view.py'
    with open(vis_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Quick 3D visualization script for Close-up-GS model
Auto-generated from view_3d_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = np.load('{files['parameters']}')
centers = data['centers']
opacities = data['opacities'].flatten()

# Filter active Gaussians
active_mask = opacities > 0.01
active_centers = centers[active_mask]
active_opacities = opacities[active_mask]

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sample for performance
if len(active_centers) > 1000:
    indices = np.random.choice(len(active_centers), 1000, replace=False)
    active_centers = active_centers[indices]
    active_opacities = active_opacities[indices]

scatter = ax.scatter(active_centers[:, 0], 
                   active_centers[:, 1], 
                   active_centers[:, 2],
                   c=active_opacities, 
                   cmap='viridis',
                   alpha=0.6,
                   s=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Close-up-GS 3D Model')

plt.colorbar(scatter, label='Opacity')
plt.tight_layout()
plt.show()
''')
    
    print(f"✓ Quick visualization script: {vis_script}")
    print("  - Run with: python quick_view.py")

def main():
    parser = argparse.ArgumentParser(description='View Close-up-GS 3D Model')
    parser.add_argument('--model_dir', type=str, default='./outputs/3d_models',
                      help='Directory containing exported 3D model files')
    parser.add_argument('--visualize', action='store_true',
                      help='Show 3D visualization using matplotlib')
    parser.add_argument('--export_info', action='store_true', default=True,
                      help='Show export information for external tools')
    
    args = parser.parse_args()
    
    models_dir = Path(args.model_dir)
    
    if not models_dir.exists():
        print(f"Error: Model directory not found: {models_dir}")
        print("Make sure you've run training and exported the 3D model")
        return
    
    # Load model data
    data, files = load_model_data(models_dir)
    
    # Print summary
    print_model_summary(data, files)
    
    # Show visualization
    if args.visualize:
        visualize_with_matplotlib(data)
    
    # Export information
    if args.export_info:
        export_for_external_tools(data, files, models_dir)
    
    print(f"\n{'='*50}")
    print("3D Model successfully loaded and analyzed!")
    print("Use --visualize flag to show 3D plot")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()















