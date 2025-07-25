#!/bin/bash
# LucidDreamer优化版环境配置脚本

# 检查是否安装了Python
if ! command -v python3 &> /dev/null
then
    echo "Python3未安装，请先安装Python3.9+"
    exit 1
fi

# 检查是否安装了pip
if ! command -v pip3 &> /dev/null
then
    echo "pip3未安装，正在安装..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv lucid_optimized_env
source lucid_optimized_env/bin/activate

# 更新pip
echo "更新pip..."
pip install --upgrade pip

# 安装PyTorch (CUDA 11.8)
echo "安装PyTorch..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录结构..."
mkdir -p inputs outputs logs pretrained

# 提示下载预训练模型
echo "请手动下载以下预训练模型并放入pretrained目录："
echo "1. ESRGAN模型: https://github.com/xinntao/ESRGAN/releases"
echo "2. ZoeDepth模型: https://github.com/isl-org/ZoeDepth"
echo "3. 3DGS基础模型: 参考LucidDreamer官方仓库"

echo "环境配置完成！"
echo "使用以下命令激活环境：source lucid_optimized_env/bin/activate"
echo "运行程序：python run_optimized.py"
