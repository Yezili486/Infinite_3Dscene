#!/usr/bin/env python3
"""
简化版Close-up GS运行脚本
使用可用的依赖包运行项目
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import json
import argparse
from models.closeup_gs import CloseupGaussianEnhancer
from utils.point_cloud_utils import generate_point_cloud
from utils.camera_utils import load_cameras

class SimplifiedCloseupGS:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化Closeup GS模型
        self.closeup_enhancer = CloseupGaussianEnhancer().to(self.device)
        
    def enhance_image(self, image_path):
        """简单的图像增强（不使用ESRGAN）"""
        img = Image.open(image_path).convert('RGB')
        # 简单的图像预处理
        img_array = np.array(img)
        # 调整图像大小
        img_resized = cv2.resize(img_array, (512, 512))
        return Image.fromarray(img_resized)
    
    def estimate_depth_simple(self, image):
        """简单的深度估计（不使用ZoeDepth）"""
        # 创建一个简单的深度图
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # 使用高斯模糊模拟深度信息
        depth = cv2.GaussianBlur(gray, (15, 15), 0)
        return depth
    
    def process_image(self, image_path, cameras_path, output_dir):
        """处理单张图像"""
        print(f"处理图像: {image_path}")
        
        # 1. 图像增强
        enhanced_img = self.enhance_image(image_path)
        print("图像增强完成")
        
        # 2. 深度估计
        depth_map = self.estimate_depth_simple(enhanced_img)
        print("深度估计完成")
        
        # 3. 生成点云
        point_cloud = generate_point_cloud(enhanced_img, depth_map)
        print("点云生成完成")
        
        # 4. Closeup GS增强
        point_cloud_tensor = torch.from_numpy(point_cloud).float().to(self.device)
        enhanced_pc = self.closeup_enhancer(point_cloud_tensor)
        print("Closeup GS增强完成")
        
        # 5. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存增强后的图像
        enhanced_img.save(os.path.join(output_dir, "enhanced_image.jpg"))
        
        # 保存深度图
        depth_img = Image.fromarray(depth_map)
        depth_img.save(os.path.join(output_dir, "depth_map.jpg"))
        
        # 保存点云数据
        enhanced_pc_np = enhanced_pc.detach().cpu().numpy()
        np.save(os.path.join(output_dir, "enhanced_point_cloud.npy"), enhanced_pc_np)
        
        # 保存相机参数
        if os.path.exists(cameras_path):
            import shutil
            shutil.copy(cameras_path, os.path.join(output_dir, "cameras.json"))
        
        print(f"结果保存到: {output_dir}")
        return output_dir

def create_sample_data():
    """创建示例数据"""
    # 创建示例图像
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sample_image_path = "sample_input.jpg"
    Image.fromarray(sample_image).save(sample_image_path)
    
    # 创建示例相机参数
    camera_params = {
        "cameras": [
            {
                "id": 0,
                "width": 512,
                "height": 512,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 256.0,
                "cy": 256.0,
                "transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            },
            {
                "id": 1,
                "width": 512,
                "height": 512,
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 256.0,
                "cy": 256.0,
                "transform": [
                    [0.707, 0.0, 0.707, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-0.707, 0.0, 0.707, 2.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
    }
    
    cameras_path = "sample_cameras.json"
    with open(cameras_path, "w") as f:
        json.dump(camera_params, f, indent=2)
    
    return sample_image_path, cameras_path

def main():
    parser = argparse.ArgumentParser(description="简化版Close-up GS")
    parser.add_argument("--input", default=None, help="输入图像路径")
    parser.add_argument("--cameras", default=None, help="相机参数JSON路径")
    parser.add_argument("--output", default="./results", help="输出目录")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    
    args = parser.parse_args()
    
    # 创建示例数据（如果没有提供输入）
    if args.create_sample or (args.input is None):
        print("创建示例数据...")
        input_path, cameras_path = create_sample_data()
        print(f"示例图像: {input_path}")
        print(f"示例相机参数: {cameras_path}")
    else:
        input_path = args.input
        cameras_path = args.cameras or "sample_cameras.json"
    
    # 初始化模型
    print("初始化Close-up GS模型...")
    model = SimplifiedCloseupGS(device=args.device)
    
    # 处理图像
    print("开始处理...")
    output_dir = model.process_image(input_path, cameras_path, args.output)
    
    print("\n" + "="*50)
    print("处理完成！")
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    print("="*50)

if __name__ == "__main__":
    main() 