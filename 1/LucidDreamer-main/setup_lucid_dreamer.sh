#!/bin/bash
# LucidDreamer环境配置脚本

echo "开始配置LucidDreamer环境..."
echo "================================="

# 创建并激活虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv lucid_dreamer_env
source lucid_dreamer_env/bin/activate
echo "虚拟环境已激活"

# 更新pip
echo "更新pip..."
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择，这里假设CUDA 11.8)
echo "安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖库
echo "安装其他依赖库..."
pip install -r requirements.txt

# 克隆LucidDreamer代码库
echo "克隆LucidDreamer代码库..."
git clone https://github.com/lucid-dreamer/lucid-dreamer.git
cd lucid-dreamer

# 安装额外依赖
echo "安装额外依赖..."
pip install -e .

# 下载预训练模型（如果有）
echo "下载预训练模型..."
mkdir -p checkpoints
cd checkpoints
# 这里需要根据实际情况添加下载链接
# wget https://example.com/lucid_dreamer_checkpoint.pth
cd ..

# 测试环境
echo "测试环境是否配置成功..."
python test_env.py

echo "================================="
echo "LucidDreamer环境配置完成！"
echo "使用 'source lucid_dreamer_env/bin/activate' 激活环境" 