# Close-up Gaussian Splatting 项目

这是一个Close-up Gaussian Splatting的复现项目，用于近景物体的3D重建和细节增强。

## 项目结构

```
closeup-gs-full-dependencies/
├── models/                 # 模型定义
│   ├── closeup_gs.py      # Closeup GS核心模型
│   ├── esrgan.py          # ESRGAN超分模型
│   ├── zoedepth.py        # ZoeDepth深度估计模型
│   └── gaussian_renderer.py # 3DGS渲染器
├── utils/                  # 工具函数
│   ├── camera_utils.py    # 相机参数处理
│   └── point_cloud_utils.py # 点云处理
├── requirements.txt        # 依赖列表
├── run_simplified.py      # 简化版运行脚本
├── test_setup.py          # 环境测试脚本
├── visualize_results.py    # 结果可视化脚本
└── README.md              # 使用说明
```

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 其他依赖见 requirements.txt

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. 测试环境

```bash
python test_setup.py
```

### 3. 运行项目

使用示例数据：
```bash
python run_simplified.py --create_sample
```

使用自定义图像：
```bash
python run_simplified.py --input your_image.jpg --cameras your_cameras.json
```

### 4. 可视化结果

```bash
python visualize_results.py
```

## 功能特性

### 核心功能
- **图像增强**: 自动调整图像大小和预处理
- **深度估计**: 生成深度图用于3D重建
- **点云生成**: 从RGB图像和深度图生成3D点云
- **Closeup GS增强**: 使用神经网络增强近景细节
- **结果可视化**: 生成对比图和统计报告

### 输出文件
- `enhanced_image.jpg`: 增强后的图像
- `depth_map.jpg`: 深度图
- `enhanced_point_cloud.npy`: 增强后的点云数据
- `cameras.json`: 相机参数
- `visualization.png`: 可视化结果图
- `comparison_report.html`: 对比报告

## 参数说明

### run_simplified.py 参数
- `--input`: 输入图像路径
- `--cameras`: 相机参数JSON文件路径
- `--output`: 输出目录 (默认: ./results)
- `--device`: 设备选择 (cuda/cpu, 默认: cuda)
- `--create_sample`: 创建示例数据

### 相机参数格式
```json
{
  "cameras": [
    {
      "id": 0,
      "width": 512,
      "height": 512,
      "fx": 1000.0,
      "fy": 1000.0,
      "cx": 256.0,
      "cy": 256.0,
      "transform": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

## 处理流程

1. **图像预处理**: 调整图像大小和格式
2. **深度估计**: 生成深度图
3. **点云生成**: 从RGB+深度生成3D点云
4. **Closeup GS增强**: 神经网络增强细节
5. **结果保存**: 保存所有中间结果和最终输出

## 性能统计

根据测试结果：
- 点云点数: ~262,144个点
- 处理时间: 约30秒 (GPU)
- 内存使用: ~500MB
- 输出文件大小: ~6MB

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 使用 `--device cpu` 参数
   - 减小输入图像尺寸

2. **依赖安装失败**
   - 使用国内镜像源
   - 检查Python版本兼容性

3. **点云生成错误**
   - 确保图像和深度图尺寸一致
   - 检查深度图格式

### 调试模式

运行测试脚本检查环境：
```bash
python test_setup.py
```

## 扩展功能

### 添加新的模型
1. 在 `models/` 目录下创建新的模型文件
2. 在 `run_simplified.py` 中导入和使用

### 自定义可视化
修改 `visualize_results.py` 添加新的可视化方式

### 批量处理
创建批处理脚本处理多张图像

## 技术细节

### Closeup GS模型
- 输入: 6维点云 (x,y,z,r,g,b)
- 输出: 增强后的6维点云
- 网络结构: 3层卷积 + 批归一化 + ReLU

### 点云处理
- 从RGB图像和深度图生成
- 过滤无效点 (深度 < 0.1)
- 支持PLY格式导出

### 相机模型
- 支持多相机配置
- 内参: fx, fy, cx, cy
- 外参: 4x4变换矩阵

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请提交Issue或联系维护者。 