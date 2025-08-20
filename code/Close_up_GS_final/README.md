# Close-up-GS: High-Quality Close-up View Synthesis

A PyTorch implementation of Close-up-GS for high-quality close-up view synthesis, optimized for RTX 3070Ti.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download LLFF Dataset (Optional)
For real-world data training, download the LLFF dataset:

**Option A: Using Python Script (Cross-platform)**
```bash
python download_llff.py
```

**Option B: Using Shell Script (Linux/Mac)**
```bash
chmod +x download_llff.sh
./download_llff.sh
```

**Option C: Using Batch File (Windows)**
```cmd
download_llff.bat
```

**Option D: Manual Download**
```bash
git clone https://github.com/Fyusion/LLFF
cd LLFF
bash download_data.sh
cd ..
mv LLFF/data data/llff/
```

### 3. Run Training

**Synthetic Dataset (Quick Test)**
```bash
python train_closeup_gs.py --data_path ./test_data --dataset_type synthetic --target_resolution 256 256 --debug --num_samples 5 --output_dir ./outputs
```

**LLFF Dataset (Real Photos)**
```bash
python train_closeup_gs.py --data_path data/llff/fern --dataset_type llff --target_resolution 256 256 --output_dir ./outputs
```

**Custom Real Photos**
```bash
python train_closeup_gs.py --data_path llff_data --dataset_type real_photos --target_resolution 256 256 --output_dir ./outputs
```

## Project Structure

```
Close_up_GS_final/
├── models/                 # Core models
│   ├── gs_model.py        # Gaussian Splatting baseline
│   └── closeup_refiner.py # Close-up view refiner
├── train/                  # Training modules
│   └── closeup_trainer.py # Main trainer
├── utils/                  # Utilities
│   ├── camera.py          # Camera operations
│   ├── metrics.py         # Evaluation metrics
│   ├── progressive_training.py # Progressive training
│   └── view_selection.py  # View selection algorithms
├── data/                   # Dataset handling
│   └── dataset.py         # Data loading
├── config/                 # Configuration files
│   ├── closeup_gs.yaml    # Main config
│   └── debug_gs.yaml      # Debug config
├── outputs/                # Training results
├── download_llff.sh       # LLFF dataset download script
├── download_llff.py       # Python download script
└── train_closeup_gs.py    # Main training script
```

## Core Features

### 1. Gaussian Splatting Baseline (GSModel)
- **3D Representation**: 200 Gaussians with position, scale, rotation, opacity, and SH coefficients
- **Rendering**: Real-time differentiable rendering
- **Optimization**: Densification and pruning strategies

### 2. Close-up View Refinement
- **See3D Integration**: View synthesis and super-resolution
- **SUPIR Enhancement**: Post-processing detail enhancement
- **Photometric Consistency**: Geometric consistency checks

### 3. Smart View Selection
- **Anchor Views**: Optimal reference view selection
- **Update Views**: Progressive view expansion
- **Distance Weighting**: Spatial-aware view scoring

### 4. Progressive Training
- **3-Round Strategy**: Baseline → Refinement → Fine-tuning
- **Self-Training**: Iterative quality improvement
- **Memory Optimization**: RTX 3070Ti optimized

## RTX 3070Ti Optimizations

- **Memory Management**: 75% GPU memory fraction
- **Mixed Precision**: Automatic mixed precision (AMP)
- **Memory Cleanup**: Aggressive cache clearing
- **Batch Optimization**: Reduced batch sizes for 8GB VRAM

## Training Flow

1. **Initial Optimization**: 10 iterations baseline training
2. **Progressive Update**: 3 rounds of refinement
3. **Fine-tuning**: 5 iterations final optimization
4. **Evaluation**: PSNR, SSIM, LPIPS metrics

## Output Results

### Training Statistics
- `training_stats.json`: Complete training metrics
- `final_model.pth`: Trained model checkpoint

### Images
- `evaluation_images/`: Final rendered results
- `baseline_results/`: Baseline model outputs
- `original_images/`: Input training images

### 3D Models
- `3d_models/gaussians_spheres.obj`: 3D model for visualization
- `3d_models/gaussian_parameters.npz`: Raw Gaussian parameters
- `3d_models/model_statistics.json`: Model statistics

## Configuration

### Debug Mode (Fast Testing)
```yaml
# config/debug_gs.yaml
training:
  baseline_iterations: 10
  refinement_iterations: 5
  finetune_iterations: 5
```

### Production Mode
```yaml
# config/closeup_gs.yaml
training:
  baseline_iterations: 30000
  refinement_iterations: 10000
  finetune_iterations: 5000
```

## Dataset Support

### Synthetic Dataset
- Programmatically generated 3D scenes
- Fast testing and debugging
- Controlled environment

### LLFF Dataset
- Real-world photos with camera poses
- Standard NeRF dataset format
- High-quality training data

### Custom Real Photos
- User-provided images
- LLFF format compatible
- Flexible data source

## Performance Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **DINO**: Feature similarity (no-reference)

## Memory Usage

- **Baseline Training**: ~4GB VRAM
- **Refinement Phase**: ~6GB VRAM
- **Fine-tuning**: ~5GB VRAM
- **Total Peak**: ~7GB VRAM

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce `target_resolution`
   - Use debug configuration
   - Check memory fraction settings

2. **Device Mismatch Errors**
   - Ensure all tensors are on CUDA
   - Check camera device placement
   - Verify model device assignment

3. **Training Stuck**
   - Check LPIPS model initialization
   - Verify data loading
   - Monitor GPU memory usage

## Citation

If you use this code, please cite the original Close-up-GS paper:

```bibtex
@article{closeup-gs-2024,
  title={Close-up-GS: High-Quality Close-up View Synthesis},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
