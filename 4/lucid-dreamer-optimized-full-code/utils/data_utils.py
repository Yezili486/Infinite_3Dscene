import cv2
import os
import torch
import numpy as np
from PIL import Image

def load_input_image(input_path):
    """
    加载输入图像并进行基础预处理
    
    参数:
        input_path: 图像路径
    
    返回:
        img: 处理后的图像 (H, W, C)，BGR格式
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入图像不存在: {input_path}")
    
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")
    
    # 确保图像是3通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img

def save_output_image(image_tensor, output_path):
    """
    保存输出图像
    
    参数:
        image_tensor: 图像张量 (C, H, W)，值在[0,1]之间
        output_path: 输出路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为numpy数组
    img_np = image_tensor.cpu().detach().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # 转换为BGR格式并保存
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)

def create_input_directory(input_dir):
    """创建输入目录并提示用户放置输入图像"""
    os.makedirs(input_dir, exist_ok=True)
    example_path = os.path.join(input_dir, "input.jpg")
    
    if not os.path.exists(example_path):
        # 创建示例图像作为占位符
        example_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        cv2.putText(
            example_img, "Replace with your input image", 
            (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.imwrite(example_path, example_img)
        print(f"请将输入图像放置在: {example_path}")
    
    return example_path

def generate_test_scenes(output_dir, num_scenes=3):
    """生成测试场景图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    scenes = [
        {"name": "cyberpunk", "desc": "赛博朋克街景"},
        {"name": "study", "desc": "室内书房"},
        {"name": "anime", "desc": "动漫角色"}
    ]
    
    for i, scene in enumerate(scenes[:num_scenes]):
        path = os.path.join(output_dir, f"{scene['name']}.jpg")
        if not os.path.exists(path):
            # 创建示例图像
            img = np.ones((512, 768, 3), dtype=np.uint8) * 240
            cv2.putText(
                img, f"Test Scene: {scene['desc']}", 
                (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (50, 50, 50), 3, cv2.LINE_AA
            )
            cv2.imwrite(path, img)
            print(f"生成测试场景图像: {path}")
