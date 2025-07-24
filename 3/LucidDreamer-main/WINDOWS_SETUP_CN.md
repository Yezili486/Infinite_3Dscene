# LucidDreamer Windows 配置指南

## 📋 前置要求

在运行配置脚本之前，请确保您的系统满足以下要求：

### 1. 安装 Python 3.9
- 从 [Python官网](https://www.python.org/downloads/windows/) 下载 Python 3.9
- **重要**: 安装时必须勾选 "Add Python to PATH"
- 验证安装: 在命令行输入 `python --version` 应显示 Python 3.9.x

### 2. 安装 Visual Studio Build Tools (必需)
- 下载 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- 安装时选择 "C++ build tools" 和 "MSVC v143 - VS 2022 C++ x64/x86 build tools"
- 这是编译CUDA扩展模块必需的

### 3. 安装 CUDA 11.8+ (推荐)
- 从 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-11-8-0-download-archive) 下载 CUDA 11.8
- 确保您的GPU支持CUDA (GTX 1060 6GB 或更高)
- 验证安装: 在命令行输入 `nvcc --version`

### 4. 安装 Git (如果需要下载GLM库)
- 从 [Git官网](https://git-scm.com/download/win) 下载并安装

## 🚀 快速开始

### 方法一: 全自动配置 (推荐)

1. **双击运行配置脚本**
   ```
   setup_luciddreamer_windows.bat
   ```
   
2. **等待配置完成**
   - 脚本会自动创建虚拟环境
   - 安装所有依赖包
   - 编译CUDA扩展
   - 运行演示实例

3. **查看结果**
   - 生成的文件保存在 `outputs/christmas_demo/` 目录
   - `.ply` 文件: 3D高斯场景文件
   - `.mp4` 文件: 渲染的视频

### 方法二: 快速演示 (环境已配置)

如果环境已经配置完成，可以使用快速演示脚本：

```
quick_demo.bat
```

选择以下演示之一：
1. **Christmas场景**: 温馨圣诞客厅
2. **Cabin场景**: 森林中的魔法师小屋  
3. **Doge场景**: 舒适的客厅环境
4. **Web界面**: 在浏览器中交互式生成

## 📁 输出文件说明

生成完成后，您会在 `outputs/` 目录下找到：

```
outputs/
├── christmas_demo/
│   ├── point_cloud/
│   │   └── point_cloud.ply      # 3D高斯场景文件
│   ├── images/                  # 生成的图片序列
│   └── videos/                  # 渲染的视频文件
│       ├── back_and_forth.mp4   # 来回移动视频
│       ├── llff.mp4             # LLFF相机路径
│       └── headbanging.mp4      # 摇头视角
```

## 🎯 查看结果

### 1. 查看3D场景 (.ply文件)
推荐使用以下在线查看器：
- **Super-Splat**: https://playcanvas.com/super-splat (推荐)
- **Splat Viewer**: https://antimatter15.com/splat/
- **Spline**: https://spline.design/

### 2. 查看视频 (.mp4文件)
直接用任何视频播放器打开 `videos/` 目录下的文件

## 🛠️ 手动运行示例

激活环境后，您可以手动运行其他示例：

```bash
# 激活虚拟环境
lucid_dreamer_env\Scripts\activate

# 运行不同的示例
python run.py --image examples/cabin.png --text examples/cabin.txt --save_dir outputs/cabin_demo
python run.py --image examples/island.png --text examples/island.txt --save_dir outputs/island_demo
python run.py --image examples/ruin.png --text examples/ruin.txt --save_dir outputs/ruin_demo

# 使用自定义参数
python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/custom_demo --seed 123 --diff_steps 30 --campath_gen rotate360 --campath_render headbanging
```

## 🌐 Web界面使用

启动Web界面进行交互式生成：

```bash
# 轻量版 (推荐开始)
python app_mini.py

# 完整版 (需要更多内存和GPU)
python app.py
```

然后在浏览器中访问: http://localhost:7860

## 🔧 常见问题

### Q: 编译错误 "Microsoft Visual C++ 14.0 is required"
**A**: 请安装 Visual Studio Build Tools，确保包含 C++ 编译器

### Q: CUDA编译失败
**A**: 
1. 确保 CUDA 版本与 PyTorch 版本匹配 (CUDA 11.8 + PyTorch 2.0.1)
2. 确保安装了正确的 Visual Studio Build Tools
3. 重启命令行后重试

### Q: "python command not found"
**A**: 
1. 重新安装 Python，确保勾选 "Add Python to PATH"
2. 重启命令行
3. 验证: `python --version`

### Q: GPU内存不足
**A**:
1. 降低 `--diff_steps` 参数 (默认20，可改为10)
2. 关闭其他占用GPU的程序
3. 使用更小的输入图片

### Q: 生成速度很慢
**A**:
1. 确保使用了GPU版本的PyTorch
2. 检查 `torch.cuda.is_available()` 返回 `True`
3. 第一次运行需要下载预训练模型

## 📊 系统要求

### 最低配置
- **CPU**: Intel i5 或 AMD Ryzen 5
- **内存**: 8GB RAM
- **GPU**: GTX 1060 6GB (支持CUDA)
- **存储**: 5GB 可用空间

### 推荐配置  
- **CPU**: Intel i7 或 AMD Ryzen 7
- **内存**: 16GB+ RAM
- **GPU**: RTX 3060 12GB 或更高
- **存储**: 10GB+ 可用空间

## 📝 参数说明

### 主要参数
- `--image`: 输入图片路径
- `--text`: 文本提示文件路径
- `--save_dir`: 输出目录
- `--seed`: 随机种子 (影响生成结果)
- `--diff_steps`: 扩散步数 (越高质量越好但越慢)

### 相机路径
- `--campath_gen`: 生成时相机路径
  - `lookdown`: 俯视
  - `lookaround`: 环视
  - `rotate360`: 360度旋转
- `--campath_render`: 渲染时相机路径  
  - `back_and_forth`: 来回移动
  - `llff`: LLFF风格
  - `headbanging`: 摇头视角

## 💡 使用技巧

### 文本提示优化
1. **室内场景**: 使用简单描述，如 "cozy livingroom", "modern kitchen"
2. **避免人物**: 不要使用 "1girl", "person" 等，会产生重复人物
3. **使用负面提示**: 通过 `--neg_text` 参数排除不想要的元素

### 图片选择
1. **清晰度**: 使用高分辨率、清晰的图片
2. **构图**: 避免过于复杂的场景
3. **光照**: 均匀光照效果更好

## 🔗 相关链接

- [项目主页](https://luciddreamer-cvlab.github.io/)
- [论文](https://arxiv.org/abs/2311.13384)
- [GitHub仓库](https://github.com/luciddreamer-cvlab/LucidDreamer)
- [在线Demo](https://huggingface.co/spaces/ironjr/LucidDreamer-mini)
- [Colab演示](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb)

---

**享受您的3D创作之旅！** 🎨✨ 