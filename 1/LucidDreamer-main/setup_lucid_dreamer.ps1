# LucidDreamer环境配置脚本 (PowerShell版本)

Write-Host "开始配置LucidDreamer环境..." -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# 创建并激活虚拟环境
Write-Host "创建Python虚拟环境..." -ForegroundColor Yellow
python -m venv lucid_dreamer_env
& ".\lucid_dreamer_env\Scripts\Activate.ps1"
Write-Host "虚拟环境已激活" -ForegroundColor Green

# 更新pip
Write-Host "更新pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择，这里假设CUDA 11.8)
Write-Host "安装PyTorch..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖库
Write-Host "安装其他依赖库..." -ForegroundColor Yellow
pip install -r requirements.txt

# 克隆LucidDreamer代码库
Write-Host "克隆LucidDreamer代码库..." -ForegroundColor Yellow
git clone https://github.com/lucid-dreamer/lucid-dreamer.git
cd lucid-dreamer

# 安装额外依赖
Write-Host "安装额外依赖..." -ForegroundColor Yellow
pip install -e .

# 下载预训练模型（如果有）
Write-Host "下载预训练模型..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path checkpoints
cd checkpoints
# 这里需要根据实际情况添加下载链接
# Invoke-WebRequest -Uri "https://example.com/lucid_dreamer_checkpoint.pth" -OutFile "lucid_dreamer_checkpoint.pth"
cd ..

# 测试环境
Write-Host "测试环境是否配置成功..." -ForegroundColor Yellow
python test_env.py

Write-Host "=================================" -ForegroundColor Green
Write-Host "LucidDreamer环境配置完成！" -ForegroundColor Green
Write-Host "使用 '.\lucid_dreamer_env\Scripts\Activate.ps1' to activate environment" -ForegroundColor Cyan 