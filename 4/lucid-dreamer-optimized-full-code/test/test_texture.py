import yaml
import cv2
import os
import torch
from models.dreamer import EnhancedDreamer
from utils.data_utils import load_input_image, save_output_image

def test_texture_enhancement(config_path):
    """测试纹理增强效果"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = os.path.join(config['paths']['output_dir'], 'texture_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化dreamer模块
    dreamer = EnhancedDreamer(config)
    
    # 加载测试图像
    test_images = [
        os.path.join(config['paths']['input_dir'], 'input.jpg'),
        os.path.join(config['paths']['input_dir'], 'test_cyberpunk.jpg')
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"跳过不存在的测试图像: {img_path}")
            continue
            
        # 加载图像
        img = load_input_image(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存原始图像
        save_output_image(
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            os.path.join(output_dir, f"{img_name}_original.jpg")
        )
        
        # 无纹理增强处理
        dreamer.use_enhance = False
        img_tensor = dreamer.preprocess_image(img)
        save_output_image(
            img_tensor[0],
            os.path.join(output_dir, f"{img_name}_no_enhance.jpg")
        )
        
        # 有纹理增强处理
        dreamer.use_enhance = True
        enhanced_tensor = dreamer.preprocess_image(img)
        save_output_image(
            enhanced_tensor[0],
            os.path.join(output_dir, f"{img_name}_enhanced.jpg")
        )
        
        print(f"纹理增强测试完成，结果保存在: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试纹理增强效果")
    parser.add_argument('--config', type=str, default='configs/lucid_optimized.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    test_texture_enhancement(args.config)
