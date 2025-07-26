# LucidDreamer Windows 安装指南

## 🎯 快速开始

### 方法一：完整自动安装（推荐新手）
```bash
# 以管理员身份运行 PowerShell 或 CMD
setup_windows.bat
```

### 方法二：简化安装（适用于有基础环境的用户）
```bash
setup_windows_simple.bat
```

## 📋 系统要求

### 必需组件
- **操作系统**: Windows 10/11 (64位)
- **Python**: 3.9.x
- **CUDA**: 11.8 
- **GPU**: NVIDIA GPU (建议 8GB+ 显存)
- **Visual Studio**: 2019/2022 Build Tools 或完整版

### 安装顺序
1. **Python 3.9** - [下载地址](https://www.python.org/downloads/release/python-3916/)
2. **CUDA 11.8** - [下载地址](https://developer.nvidia.com/cuda-11-8-0-download-archive)
3. **Visual Studio Build Tools** - [下载地址](https://visualstudio.microsoft.com/downloads/)
   - 选择 "C++ build tools"
   - 确保安装 "MSVC v143 - VS 2022 C++ x64/x86 build tools"

## 🚀 安装步骤

### 步骤 1: 准备环境
1. 以**管理员身份**打开 PowerShell 或命令提示符
2. 导航到项目目录
3. 运行安装脚本

### 步骤 2: 运行安装脚本
```bash
# 完整安装（包含所有检查和错误处理）
setup_windows.bat

# 或者简化安装（适用于已有基础环境）
setup_windows_simple.bat
```

### 步骤 3: 启动程序
安装完成后，可以使用以下方式启动：

```bash
# 方式 1: 使用生成的启动脚本
start_lucid_dreamer.bat

# 方式 2: 手动激活环境后运行
lucid_optimized_env\Scripts\activate.bat
python run_optimized.py

# 方式 3: 启动 Web 界面
start_app.bat          # 完整版 Web 应用
start_mini_app.bat     # 简化版 Web 应用
```

## 🔧 故障排除

### 运行诊断脚本
```bash
troubleshoot_windows.bat
```

### 常见问题及解决方案

#### 1. CUDA 编译错误
**症状**: `nvcc: command not found` 或编译失败
**解决方案**:
- 确保 CUDA 11.8 已正确安装
- 检查环境变量 `CUDA_PATH` 是否设置
- 重新安装 Visual Studio Build Tools

#### 2. Visual Studio Build Tools 问题
**症状**: `error: Microsoft Visual C++ 14.0 is required`
**解决方案**:
- 安装 Visual Studio 2019/2022 Build Tools
- 确保选择了 C++ 构建工具
- 重启系统后重新运行安装脚本

#### 3. Python 版本不兼容
**症状**: `Python 3.9 required` 错误
**解决方案**:
- 安装 Python 3.9.x
- 确保 `python --version` 显示 3.9.x

#### 4. GPU 内存不足
**症状**: `CUDA out of memory`
**解决方案**:
- 关闭其他占用 GPU 的程序
- 降低渲染分辨率
- 使用较小的输入图像

#### 5. 网络连接问题
**症状**: 下载超时或失败
**解决方案**:
- 检查网络连接
- 使用 VPN 或代理
- 手动下载所需文件

## 📁 目录结构

安装完成后的目录结构：
```
lucid-dreamer-optimized-full-code/
├── lucid_optimized_env/           # 虚拟环境
├── inputs/                        # 输入文件
├── outputs/                       # 输出结果
├── logs/                         # 日志文件
├── pretrained/                   # 预训练模型
├── setup_windows.bat             # 完整安装脚本
├── setup_windows_simple.bat      # 简化安装脚本
├── troubleshoot_windows.bat      # 故障排除脚本
├── start_lucid_dreamer.bat       # 主程序启动脚本
├── start_app.bat                 # Web 应用启动脚本
└── start_mini_app.bat            # 简化 Web 应用启动脚本
```

## 🎮 使用说明

### 命令行模式
```bash
# 激活环境
lucid_optimized_env\Scripts\activate.bat

# 基本用法
python run_optimized.py --input examples/cabin.png --text examples/cabin.txt

# 高质量渲染
python run_optimized.py --input examples/cabin.png --text examples/cabin.txt --quality high

# 自定义输出
python run_optimized.py --input examples/cabin.png --text examples/cabin.txt --output my_output
```

### Web 界面模式
```bash
# 启动完整版 Web 界面
start_app.bat

# 启动简化版 Web 界面（更快启动）
start_mini_app.bat
```

## 📋 性能优化建议

### GPU 内存优化
- 对于 8GB GPU: 使用默认设置
- 对于 6GB GPU: 降低渲染分辨率至 512x512
- 对于 4GB GPU: 使用简化模式

### 渲染质量设置
- **快速预览**: `--quality fast`
- **标准质量**: `--quality normal` (默认)
- **高质量**: `--quality high`

## 🆘 获取帮助

1. **运行诊断脚本**: `troubleshoot_windows.bat`
2. **查看日志文件**: `logs/` 目录下的日志文件
3. **检查 GPU 状态**: 任务管理器 → 性能 → GPU
4. **测试安装**: 运行故障排除脚本中的测试命令

## 📚 更多信息

- [原始项目文档](README.md)
- [配置文件说明](configs/lucid_optimized.yaml)
- [示例文件](examples/)

---

**注意**: 首次运行时会自动下载预训练模型，请确保网络连接稳定。 