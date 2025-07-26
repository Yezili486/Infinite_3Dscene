# LucidDreamer优化版环境配置脚本 - PowerShell版本

Write-Host "=== LucidDreamer 优化版环境配置 ===" -ForegroundColor Green

# 检查PowerShell执行策略
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "检测到PowerShell执行策略为Restricted，正在设置为RemoteSigned..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Write-Host "执行策略已更新" -ForegroundColor Green
    } catch {
        Write-Host "无法更改执行策略，请以管理员身份运行PowerShell并执行: Set-ExecutionPolicy RemoteSigned" -ForegroundColor Red
        Read-Host "按任意键退出"
        exit 1
    }
}

# 检查Python安装
Write-Host "正在检查Python安装..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "发现Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python未安装或未添加到PATH" -ForegroundColor Red
    Write-Host "请从以下地址下载并安装Python 3.9+: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "安装时请确保勾选 'Add Python to PATH' 选项" -ForegroundColor Yellow
    Read-Host "按任意键退出"
    exit 1
}

# 检查pip
Write-Host "正在检查pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "发现pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "pip未找到，请重新安装Python" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

# 创建虚拟环境
Write-Host "创建虚拟环境..." -ForegroundColor Yellow
if (Test-Path "lucid_optimized_env") {
    Write-Host "虚拟环境已存在，跳过创建" -ForegroundColor Green
} else {
    python -m venv lucid_optimized_env
    if ($LASTEXITCODE -ne 0) {
        Write-Host "创建虚拟环境失败" -ForegroundColor Red
        Read-Host "按任意键退出"
        exit 1
    }
}

# 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Yellow
& ".\lucid_optimized_env\Scripts\Activate.ps1"

# 更新pip
Write-Host "更新pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安装PyTorch
Write-Host "安装PyTorch (CUDA 11.8)..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟时间..." -ForegroundColor Cyan
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 检查requirements.txt是否存在
if (Test-Path "requirements.txt") {
    Write-Host "安装项目依赖..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    Write-Host "未找到requirements.txt文件" -ForegroundColor Yellow
}

# 创建目录结构
Write-Host "创建项目目录结构..." -ForegroundColor Yellow
$directories = @("inputs", "outputs", "logs", "pretrained")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Green
    } else {
        Write-Host "目录已存在: $dir" -ForegroundColor Yellow
    }
}

# 完成信息
Write-Host "`n=== 配置完成 ===" -ForegroundColor Green
Write-Host "请手动下载以下预训练模型并放入pretrained目录：" -ForegroundColor Cyan
Write-Host "1. ESRGAN模型: https://github.com/xinntao/ESRGAN/releases" -ForegroundColor White
Write-Host "2. ZoeDepth模型: https://github.com/isl-org/ZoeDepth" -ForegroundColor White
Write-Host "3. 3DGS基础模型: 参考LucidDreamer官方仓库" -ForegroundColor White

Write-Host "`n=== 使用方法 ===" -ForegroundColor Green
Write-Host "1. 激活虚拟环境:" -ForegroundColor Yellow
Write-Host "   .\lucid_optimized_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. 运行程序:" -ForegroundColor Yellow
Write-Host "   python run_optimized.py" -ForegroundColor White

Read-Host "`n按任意键退出" 