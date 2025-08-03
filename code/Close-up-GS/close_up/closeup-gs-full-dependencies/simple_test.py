#!/usr/bin/env python3
"""
简单的环境测试脚本
"""

import sys
import torch
import numpy as np
import cv2
from PIL import Image

def test_basic_imports():
    """测试基础导入"""
    print("基础库导入成功")
    print(f"  - Python版本: {sys.version}")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")

def test_models():
    """测试模型导入"""
    try:
        from models.closeup_gs import CloseupGaussianEnhancer
        print("CloseupGaussianEnhancer 导入成功")
    except ImportError as e:
        print(f"CloseupGaussianEnhancer 导入失败: {e}")
    
    try:
        from models.esrgan import ESRGANEnhancer
        print("ESRGANEnhancer 导入成功")
    except ImportError as e:
        print(f"ESRGANEnhancer 导入失败: {e}")
    
    try:
        from models.zoedepth import ZoeDepthEstimator
        print("ZoeDepthEstimator 导入成功")
    except ImportError as e:
        print(f"ZoeDepthEstimator 导入失败: {e}")

def test_utils():
    """测试工具函数"""
    try:
        from utils.point_cloud_utils import generate_point_cloud
        print("点云工具函数导入成功")
    except ImportError as e:
        print(f"点云工具函数导入失败: {e}")
    
    try:
        from utils.camera_utils import load_cameras
        print("相机工具函数导入成功")
    except ImportError as e:
        print(f"相机工具函数导入失败: {e}")

def main():
    print("=" * 50)
    print("Close-up GS 环境测试")
    print("=" * 50)
    
    # 测试基础导入
    test_basic_imports()
    print()
    
    # 测试模型导入
    test_models()
    print()
    
    # 测试工具函数
    test_utils()
    print()
    
    print("=" * 50)
    print("环境测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main() 