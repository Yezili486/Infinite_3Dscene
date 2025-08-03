#!/usr/bin/env python3
"""
测试Close-up GS环境设置
"""

import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def test_basic_imports():
    """测试基础导入"""
    print("✓ 基础库导入成功")
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
        print("✓ CloseupGaussianEnhancer 导入成功")
    except ImportError as e:
        print(f"✗ CloseupGaussianEnhancer 导入失败: {e}")
    
    try:
        from models.esrgan import ESRGANEnhancer
        print("✓ ESRGANEnhancer 导入成功")
    except ImportError as e:
        print(f"✗ ESRGANEnhancer 导入失败: {e}")
    
    try:
        from models.zoedepth import ZoeDepthEstimator
        print("✓ ZoeDepthEstimator 导入成功")
    except ImportError as e:
        print(f"✗ ZoeDepthEstimator 导入失败: {e}")

def test_utils():
    """测试工具函数"""
    try:
        from utils.point_cloud_utils import generate_point_cloud
        print("✓ 点云工具函数导入成功")
    except ImportError as e:
        print(f"✗ 点云工具函数导入失败: {e}")
    
    try:
        from utils.camera_utils import load_cameras
        print("✓ 相机工具函数导入成功")
    except ImportError as e:
        print(f"✗ 相机工具函数导入失败: {e}")

def create_dummy_data():
    """创建测试数据"""
    # 创建一个简单的测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_image_path = "test_image.jpg"
    Image.fromarray(test_image).save(test_image_path)
    print(f"✓ 创建测试图像: {test_image_path}")
    
    # 创建简单的相机参数
    camera_params = {
        "cameras": [
            {
                "id": 0,
                "width": 256,
                "height": 256,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 128.0,
                "cy": 128.0,
                "transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
    }
    
    import json
    with open("test_cameras.json", "w") as f:
        json.dump(camera_params, f, indent=2)
    print("✓ 创建测试相机参数: test_cameras.json")
    
    return test_image_path, "test_cameras.json"

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
    
    # 创建测试数据
    test_image_path, test_cameras_path = create_dummy_data()
    print()
    
    print("=" * 50)
    print("环境测试完成！")
    print("=" * 50)
    
    return test_image_path, test_cameras_path

if __name__ == "__main__":
    main() 