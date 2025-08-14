# Close-up-GS: 高质量近距离视图合成

基于论文 "Close-up-GS" 的完整 PyTorch 实现，专为 **NVIDIA RTX 3070Ti** 优化。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- NVIDIA RTX 3070Ti/4090 (8GB+ VRAM)
- CUDA 11.8+

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速训练
```bash
# 合成数据训练（推荐用于测试）
python train_closeup_gs.py --data_path ./test_data --dataset_type synthetic --target_resolution 256 256 --debug

# 真实数据训练
python train_closeup_gs.py --data_path /path/to/dataset --dataset_type lerf --target_resolution 512 512
```

## 📁 项目结构

```
Close_up_GS_final/
├── data/                    # 数据加载模块
│   ├── dataset.py          # CloseUpDataset类 (支持LERF/LLFF/NeRF)
│   └── __init__.py
├── models/                  # 模型定义
│   ├── gs_model.py         # GSModel (高斯散射基线)
│   ├── closeup_refiner.py  # CloseupRefiner (See3D+SUPIR)
│   ├── see3d_integration.py
│   └── __init__.py
├── train/                   # 训练模块
│   ├── gs_trainer.py       # GSTrainer
│   ├── closeup_trainer.py  # CloseupGSTrainer
│   └── __init__.py
├── utils/                   # 工具模块
│   ├── view_selection.py   # 智能视图选择算法
│   ├── progressive_training.py  # 渐进自训练
│   ├── config.py           # 配置管理
│   ├── logger.py           # 日志系统
│   ├── metrics.py          # 评估指标
│   ├── camera.py           # 相机模型
│   └── colmap_parser.py    # COLMAP数据解析
├── config/                  # 配置文件
│   ├── default.yaml        # 默认配置
│   ├── debug_gs.yaml       # 调试配置
│   └── gs_baseline.yaml    # 基线配置
├── docs/                    # 文档
│   └── STEP7_FINAL_SUMMARY.md
├── train_closeup_gs.py     # 主训练脚本
├── run_training.py         # 便捷运行脚本
├── run_memory_optimized.py # 内存优化版本
├── view_3d_model.py        # 3D模型可视化
├── main.py                 # 项目入口
└── requirements.txt        # 依赖列表
```

## 🎯 核心功能

### 完整实现
- ✅ **数据模块**: LERF/LLFF/NeRF数据集支持
- ✅ **GSModel**: 高斯散射基线模型
- ✅ **CloseupRefiner**: See3D+SUPIR集成
- ✅ **智能视图选择**: 锚视图和待更新视图选择
- ✅ **渐进自训练**: 多轮训练优化
- ✅ **评估系统**: PSNR/SSIM/LPIPS/DINO指标
- ✅ **3D模型导出**: PLY/OBJ/参数导出

### RTX 3070Ti 优化
- 内存管理优化 (8GB VRAM)
- 自动混合精度 (AMP)
- 批量大小优化 (batch_size=1)
- 内存清理策略

### 训练流程
1. **基线训练**: 高斯散射模型优化
2. **渐进自训练**: 3轮迭代优化
3. **端到端微调**: 最终模型优化
4. **评估导出**: 指标计算和3D模型导出

## 📊 输出结果

训练完成后，在 `outputs/` 目录下会生成：
- `baseline_results/`: 基线模型渲染结果
- `evaluation_images/`: 评估图像对比
- `original_images/`: 原始训练图像
- `3d_models/`: 3D高斯模型文件
  - `gaussians_pointcloud.ply`: PLY点云文件
  - `gaussian_parameters.npz`: 高斯参数
  - `model_statistics.json`: 模型统计信息
- `training_stats.json`: 训练统计
- `final_model.pth`: 最终模型检查点

## 🔧 配置说明

主要配置文件：
- `config/debug_gs.yaml`: 快速测试配置（减少迭代次数）
- `config/gs_baseline.yaml`: 标准训练配置
- `config/default.yaml`: 默认配置

## 📝 使用示例

```bash
# 1. 快速测试（合成数据）
python train_closeup_gs.py --data_path ./test_data --dataset_type synthetic --target_resolution 256 256 --debug --config config/debug_gs.yaml

# 2. 标准训练（真实数据）
python train_closeup_gs.py --data_path /path/to/lerf --dataset_type lerf --target_resolution 512 512 --config config/gs_baseline.yaml

# 3. 内存优化版本（8GB GPU）
python run_memory_optimized.py

# 4. 查看3D模型
python view_3d_model.py --model_path outputs/3d_models/gaussians_pointcloud.ply
```

## 📄 许可证

本项目基于论文 "Close-up-GS" 实现，仅供学术研究使用。
