@echo off
echo =====================================
echo LucidDreamer 离线快速启动
echo =====================================
echo.

REM 激活虚拟环境
if exist "lucid_dreamer_env\Scripts\activate.bat" (
    call lucid_dreamer_env\Scripts\activate.bat
    echo [信息] 虚拟环境已激活
) else (
    echo [错误] 虚拟环境不存在，请先运行配置脚本
    pause
    exit /b 1
)

echo [信息] 🎯 离线模式 - 无需下载大型模型文件
echo [信息] 使用本地examples进行3D场景生成
echo.

REM 创建输出目录
if not exist "outputs" mkdir outputs

echo 可选的演示案例:
echo 1. Christmas 圣诞客厅场景 (快速)
echo 2. Cabin 魔法师小屋场景 (中等)
echo 3. Doge 温馨客厅场景 (快速)
echo 4. Island 神秘岛屿场景 (复杂)
echo 5. Ruin 古老废墟场景 (复杂)
echo 6. 自定义输入
echo.

set /p choice="请选择要生成的场景 (1-6): "

if "%choice%"=="1" (
    set "image_file=examples/christmas.png"
    set "text_file=examples/christmas.txt"
    set "output_dir=outputs/christmas_offline"
    set "scene_name=Christmas"
) else if "%choice%"=="2" (
    set "image_file=examples/cabin.png"
    set "text_file=examples/cabin.txt"
    set "output_dir=outputs/cabin_offline"
    set "scene_name=Cabin"
) else if "%choice%"=="3" (
    set "image_file=examples/doge.png"
    set "text_file=examples/doge.txt"
    set "output_dir=outputs/doge_offline"
    set "scene_name=Doge"
) else if "%choice%"=="4" (
    set "image_file=examples/island.png"
    set "text_file=examples/island.txt"
    set "output_dir=outputs/island_offline"
    set "scene_name=Island"
) else if "%choice%"=="5" (
    set "image_file=examples/ruin.png"
    set "text_file=examples/ruin.txt"
    set "output_dir=outputs/ruin_offline"
    set "scene_name=Ruin"
) else if "%choice%"=="6" (
    echo.
    echo [信息] 自定义模式
    set /p image_file="请输入图片路径 (例: examples/christmas.png): "
    set /p text_file="请输入文本文件路径 (例: examples/christmas.txt): "
    set /p output_dir="请输入输出目录 (例: outputs/custom): "
    set "scene_name=Custom"
) else (
    echo [错误] 无效选择
    pause
    exit /b 1
)

echo.
echo =====================================
echo 🚀 开始生成 %scene_name% 场景
echo =====================================
echo.
echo 📁 输入图片: %image_file%
echo 📝 文本提示: %text_file%
echo 💾 输出目录: %output_dir%
echo.

REM 检查输入文件是否存在
if not exist "%image_file%" (
    echo [错误] 图片文件不存在: %image_file%
    pause
    exit /b 1
)

if not exist "%text_file%" (
    echo [错误] 文本文件不存在: %text_file%
    pause
    exit /b 1
)

echo [信息] 正在生成3D场景...
echo 注意: 离线模式可能需要更长时间，请耐心等待
echo.

REM 使用较低的参数以提高离线模式的成功率
python run.py --image "%image_file%" --text "%text_file%" --save_dir "%output_dir%" --seed 42 --diff_steps 10 --campath_gen lookaround --campath_render back_and_forth

if %errorlevel% neq 0 (
    echo.
    echo [警告] 生成过程中遇到问题，尝试使用更简化的参数...
    echo.
    
    REM 如果失败，尝试更简化的参数
    python run.py --image "%image_file%" --text "%text_file%" --save_dir "%output_dir%_simple" --seed 42 --diff_steps 5
    
    if %errorlevel% neq 0 (
        echo [错误] 生成失败！可能需要网络下载预训练模型
        echo.
        echo 🌐 替代方案:
        echo 1. 使用在线版本: https://huggingface.co/spaces/ironjr/LucidDreamer-mini
        echo 2. 运行网络修复脚本: fix_network_download.bat
        echo 3. 检查CUDA环境: fix_cuda_environment.bat
        pause
        exit /b 1
    ) else (
        set "output_dir=%output_dir%_simple"
    )
)

echo.
echo =====================================
echo ✅ 生成完成！
echo =====================================
echo.

echo 📁 输出文件保存在: %output_dir%
echo.

REM 检查生成的文件
if exist "%output_dir%" (
    echo 📋 生成的文件:
    dir "%output_dir%" /s /b | findstr /i "\.ply \.mp4 \.png"
    echo.
    
    echo 🎯 查看结果:
    if exist "%output_dir%\point_cloud\point_cloud.ply" (
        echo ✓ 3D场景文件: %output_dir%\point_cloud\point_cloud.ply
        echo   使用Super-Splat查看: https://playcanvas.com/super-splat
    )
    
    if exist "%output_dir%\videos" (
        echo ✓ 视频文件目录: %output_dir%\videos\
        echo   直接用视频播放器观看
    )
    
    echo.
    echo 🎨 其他示例命令:
    echo python run.py --image examples/girl.jpg --text examples/girl.txt --save_dir outputs/girl_test
    echo python run.py --image examples/elf.png --text examples/elf.txt --save_dir outputs/elf_test
) else (
    echo [警告] 输出目录不存在，可能生成失败
)

echo.
echo 💡 性能提示:
echo - 首次运行需要下载预训练模型 (需要网络)
echo - 建议GPU内存 8GB+ 以获得最佳性能
echo - 可以调整 --diff_steps 参数来平衡质量和速度
echo.

pause 