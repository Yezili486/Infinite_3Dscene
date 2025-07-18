#!/usr/bin/env python3
"""
Quick launcher for Generative Powers of Ten Gradio Demo

Usage:
    python run_gradio_demo.py
"""

import sys
import subprocess
import importlib.util

def check_and_install_dependencies():
    """检查并安装依赖包"""
    required_packages = [
        'gradio',
        'torch', 
        'torchvision',
        'opencv-python',
        'pillow',
        'numpy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 缺失")
    
    if missing_packages:
        print(f"\n🔧 正在安装缺失的包: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ 依赖包安装完成")
        except subprocess.CalledProcessError:
            print("❌ 依赖包安装失败，请手动运行:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """主函数"""
    print("=== Generative Powers of Ten - Quick Launcher ===")
    
    # 检查依赖
    print("\n📦 检查依赖包...")
    if not check_and_install_dependencies():
        return
    
    # 启动主程序
    print("\n🚀 启动Gradio界面...")
    try:
        from gradio_powers_of_ten_demo import main as demo_main
        demo_main()
    except ImportError:
        print("❌ 无法导入主程序文件 'gradio_powers_of_ten_demo.py'")
        print("请确保文件在同一目录下")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 