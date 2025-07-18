# Generative Powers of Ten - Gradio 交互界面

这是基于论文 "Generative Powers of Ten" 的交互式演示界面，使用 Gradio 构建。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install gradio torch torchvision opencv-python pillow numpy matplotlib
```

### 2. 运行界面

```bash
python gradio_powers_of_ten_demo.py
```

### 3. 访问界面

在浏览器中打开：`http://localhost:7860`

## 🎯 功能特性

### ✨ 主要功能

- **多尺度缩放栈生成**：基于文本提示生成不同缩放级别的图像
- **实时图像浏览**：使用滑块实时切换不同缩放级别
- **缩放动画**：生成从粗到细的缩放动画GIF
- **多种示例**：宇宙、自然、海洋等主题示例

### 🖼️ 界面组件

1. **左侧控制面板**：
   - 文本提示输入（支持多行或逗号分隔）
   - 缩放因子设置（必须是2的幂序列）
   - 快速示例按钮
   - 生成状态显示

2. **右侧显示面板**：
   - 主图像显示区域
   - 缩放级别滑块控制
   - 尺度信息显示
   - 动画生成控制

## 📝 使用说明

### 输入格式

**文本提示**：
```
distant galaxy with swirling spiral arms
star system with multiple planets
planet surface with mountains
close-up of alien vegetation
```

**缩放因子**：
```
1, 2, 4, 8
```

### 支持的关键词

**🌌 宇宙系列**：
- galaxy, spiral, cosmic, space
- star, system, planet
- 生成螺旋星系和行星系统图案

**🌳 自然系列**：
- surface, mountain, terrain, landscape
- tree, forest, branch, bark, insect
- 生成地形和植物图案

**🌊 海洋系列**：
- ocean, water, wave, sea
- coral, reef, underwater
- fish, tropical
- 生成海洋和水下生物图案

### 操作步骤

1. **输入提示**：在左侧输入框中填写文本提示
2. **设置缩放因子**：输入2的幂序列，如 1, 2, 4, 8
3. **生成缩放栈**：点击"🚀 生成缩放栈"按钮
4. **浏览图像**：使用滑块查看不同缩放级别
5. **创建动画**：点击"▶️ 创建缩放动画"生成GIF

## 🔧 技术实现

### 核心算法

- **算法1**：多尺度图像渲染（Pi_image, Pi_noise）
- **缩放栈数据结构**：多层图像存储和管理
- **多分辨率融合**：跨尺度信息整合

### 图案生成

演示版本使用程序化图案生成：
- 基于数学函数创建艺术图案
- 根据关键词生成不同主题的视觉效果
- 支持多尺度细节层次

### 性能优化

- 支持CPU和GPU加速
- 异步进度监控
- 内存高效的张量操作

## 📁 文件结构

```
gradio_powers_of_ten_demo.py    # 主程序文件
README_gradio.md                # 使用说明
zoom_animation.gif              # 生成的动画文件（运行后）
```

## 🎨 示例效果

### 宇宙缩放序列
```
提示: "distant galaxy → star system → planet surface → alien vegetation"
缩放: 1x → 2x → 4x → 8x
效果: 从星系视角逐步放大到表面细节
```

### 自然缩放序列
```
提示: "vast forest → tree branches → bark texture → insect detail"  
缩放: 1x → 2x → 4x → 8x
效果: 从森林全景放大到昆虫微观细节
```

### 海洋缩放序列
```
提示: "ocean from space → surface waves → coral reef → tropical fish"
缩放: 1x → 2x → 4x → 8x  
效果: 从太空海洋视角深入到鱼类特写
```

## ⚙️ 高级配置

### 自定义参数

在代码中可以调整：

```python
# 图像分辨率
H, W = 512, 512  # 可修改为其他尺寸

# 生成步数
T = 15  # 减少可提高速度，增加可提高质量

# 设备选择
device = "cuda"  # 或 "cpu"
```

### 扩展功能

1. **添加新的图案类型**：
   - 在 `create_artistic_pattern()` 函数中添加新的关键词判断
   - 实现对应的数学图案生成逻辑

2. **集成真实AI模型**：
   - 替换 `joint_multi_scale_sampling_demo()` 为完整的Stable Diffusion版本
   - 需要安装 `diffusers` 和相关模型

3. **导出更多格式**：
   - 修改动画生成支持MP4等视频格式
   - 添加高分辨率图像导出选项

## 🐛 故障排除

### 常见问题

**Q: 界面无法启动**
```bash
# 检查依赖安装
pip list | grep -E "(gradio|torch)"

# 重新安装
pip install --upgrade gradio torch
```

**Q: 生成速度慢**
```bash
# 检查设备
python -c "import torch; print(torch.cuda.is_available())"

# 减少图像尺寸或生成步数
```

**Q: 内存不足**
```bash
# 减小图像尺寸
H, W = 256, 256  # 在代码中修改

# 或者减少缩放层数
zoom_factors = [1, 2, 4]  # 而不是 [1, 2, 4, 8]
```

### 日志查看

运行时控制台会显示：
- 设备信息（CPU/GPU）
- 生成进度
- 错误信息

## 🔗 相关资源

- [原始论文](https://arxiv.org/abs/2312.02149): "Generative Powers of Ten"
- [Gradio官方文档](https://gradio.app/docs/)
- [PyTorch官方文档](https://pytorch.org/docs/)

## 📄 许可证

本项目仅用于学术研究和教育目的。请遵守相关论文和依赖库的许可协议。

---

**享受探索多尺度生成的乐趣！** 🔍✨ 