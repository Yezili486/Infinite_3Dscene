# Close-up-GS: Progressive 3D Gaussian Splatting for Close-up View Synthesis

Implementation of "Close-up-GS" paper (arXiv:2503.09396v1) - Progressive 3D Gaussian Splatting for synthesizing high-quality close-up views from distant training observations.

## Overview

This implementation reproduces the key innovations from the Close-up-GS paper:

1. **Baseline 3DGS** with anisotropic Gaussians and tile-based rasterization
2. **See3D Integration** using Stable Diffusion inpainting with multi-reference conditioning
3. **Progressive Expansion** with anchor/frontier view selection and trust region growth
4. **Fine-tuning Strategy** with densification and geometric consistency constraints

## Key Features

### 🎯 Core Innovations
- **Progressive View Expansion**: Iteratively expand from distant training views to close-up targets (3x, 9x, 27x closer)
- **Anchor View Selection**: Greedy optimization to maximize coverage while minimizing redundancy
- **See3D Proxy**: Stable Diffusion inpainting with reference-guided warping for unknown regions
- **Densification Strategy**: Adaptive Gaussian splitting in close-up regions

### 📊 Evaluation Metrics
- **Standard Metrics**: PSNR, SSIM, LPIPS
- **Perceptual Quality**: DINO score correlation with reference images
- **No-Reference Quality**: MetaIQA proxy for image quality assessment

### 🚀 Robust Implementation
- Fallback mechanisms when advanced libraries are unavailable
- Comprehensive error handling and graceful degradation
- Modular design for easy extension and customization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Close-up-GS_2

# Install required packages
pip install -r requirements.txt

# Optional: Install additional packages for full functionality
pip install diffusers transformers accelerate lpips
```

## Usage

### Basic Usage

```bash
python main.py --dataset llff --scene flower --rounds 3 --target_pos 0.0 0.0 0.0
```

### Advanced Usage

```bash
python main.py \
    --dataset llff \
    --scene flower \
    --rounds 3 \
    --target_pos 0.0 0.0 0.0 \
    --baseline_iterations 5000 \
    --finetune_iterations 3000 \
    --output_dir ./results/flower_experiment
```

### Parameters

- `--dataset`: Dataset type (`llff`)
- `--scene`: Scene name (e.g., `flower`, `fern`, `leaves`)
- `--rounds`: Number of progressive rounds (default: 3)
- `--target_pos`: Target object center coordinates (x, y, z)
- `--baseline_iterations`: Training iterations for baseline 3DGS (default: 3000)
- `--finetune_iterations`: Fine-tuning iterations per round (default: 2000)
- `--output_dir`: Output directory for results (default: `./output`)

## Architecture

### Core Components

1. **Gaussian3D**: 3D Gaussian representation with SH coefficients
2. **GaussianRenderer**: Tile-based rasterization and alpha blending
3. **See3DProxy**: Inpainting with multi-reference conditioning
4. **ViewSelector**: Anchor/frontier view selection algorithms
5. **ProgressiveGaussianSplatting**: Main orchestrator class
6. **EvaluationMetrics**: Comprehensive quality assessment

### Progressive Algorithm

```
For each round (3x, 9x, 27x closer):
  1. Select anchor views from known views (greedy optimization)
  2. Place frontier views along anchor-to-object lines
  3. Sample random views between anchors and frontiers
  4. Select views to update (distance-weighted selection)
  5. Render with current 3DGS
  6. Identify reliable/unreliable regions
  7. Refine unreliable regions with See3D proxy
  8. Apply super-resolution enhancement
  9. Fine-tune 3DGS on expanded dataset with densification
  10. Add refined views to known set
```

## Implementation Details

### 3D Gaussian Splatting
- **Representation**: Position, anisotropic covariance, opacity, SH coefficients
- **Rendering**: Perspective projection with 2D covariance computation
- **Optimization**: Adam optimizer with learning rate scheduling
- **Densification**: Gradient-based splitting of large/active Gaussians

### See3D Proxy
- **Reliable Region Detection**: Geometric consistency via depth/color comparison
- **Inpainting**: Stable Diffusion with multi-reference conditioning
- **Warping**: Simplified 3D warping (placeholder for complex geometric warping)
- **Super-Resolution**: Bicubic upsampling with sharpening filters

### View Selection
- **Coverage Scoring**: Pixel overlap computation with frontier views
- **Similarity Matrix**: Camera pose and position-based similarity
- **Greedy Optimization**: Maximize coverage while minimizing redundancy
- **Distance Discounting**: Avoid large spatial jumps in view selection

## Output Structure

```
output/
├── baseline/
│   └── baseline_comparison.png     # Training vs baseline close-up
├── round_0/                        # Training views
│   ├── view_0.png
│   └── ...
├── round_1/                        # First progressive round
│   ├── view_0.png
│   └── ...
├── round_2/                        # Second progressive round
│   └── ...
├── final_comparison.png            # Training vs baseline vs final
└── evaluation_results.json        # Quantitative metrics
```

## Evaluation Results

The implementation provides comprehensive evaluation:

- **PSNR/SSIM/LPIPS**: When ground truth is available
- **DINO Score**: Perceptual similarity with reference images
- **MetaIQA**: No-reference image quality assessment
- **Progressive Metrics**: Quality improvement across rounds

## Limitations and Notes

1. **See3D Proxy**: Uses Stable Diffusion inpainting as proxy (original See3D model not publicly available)
2. **Simplified Warping**: Complex 3D warping replaced with simplified geometric transforms
3. **Dataset**: Currently supports synthetic LLFF-style datasets
4. **Performance**: Real-time rendering not optimized (research-focused implementation)

## Dependencies

### Required
- PyTorch >= 1.9.0
- NumPy, SciPy, Matplotlib
- OpenCV, scikit-image, scikit-learn

### Optional (for full functionality)
- diffusers (Stable Diffusion inpainting)
- lpips (perceptual loss)
- transformers, accelerate

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce image resolution or batch size
2. **Diffusers not available**: Falls back to simple interpolation inpainting
3. **LPIPS import error**: Skips LPIPS metric, continues with other metrics
4. **Slow convergence**: Increase training iterations or adjust learning rates

### Performance Tips

1. Use GPU for faster training and rendering
2. Reduce image resolution for faster processing
3. Adjust number of Gaussians based on scene complexity
4. Use fewer progressive rounds for quick prototyping

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{closeupgs2024,
  title={Close-up-GS: Progressive 3D Gaussian Splatting for Close-up View Synthesis},
  author={[Authors]},
  journal={arXiv preprint arXiv:2503.09396v1},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper for licensing terms.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers. 