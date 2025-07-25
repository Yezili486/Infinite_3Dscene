import yaml
import os
import torch
import cv2
from models.dreamer import EnhancedDreamer
from models.alignment import DepthEnhancedAlignment
from models.renderer import GaussianModel
from utils.data_utils import load_input_image, save_output_image
from utils.metrics import evaluate_model

def test_occlusion_handling(config_path):
    """测试遮挡处理效果"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = os.path.join(config['paths']['output_dir'], 'occlusion_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化组件
    dreamer = EnhancedDreamer(config)
    gaussian_model = GaussianModel(config)
    alignment = DepthEnhancedAlignment(config, gaussian_model)
    
    # 加载含遮挡的测试图像
    test_image_path = os.path.join(config['paths']['input_dir'], 'test_occlusion.jpg')
    if not os.path.exists(test_image_path):
        # 创建测试图像（含遮挡）
        img = cv2.imread(os.path.join(config['paths']['input_dir'], 'input.jpg'))
        if img is None:
            # 创建示例遮挡图像
            img = np.ones((512, 768, 3), dtype=np.uint8) * 240
            # 画一个前景物体
            cv2.rectangle(img, (200, 200), (400, 400), (0, 0, 255), -1)
            # 画一个背景物体
            cv2.circle(img, (500, 300), 100, (0, 255, 0), -1)
            cv2.imwrite(test_image_path, img)
        else:
            # 在现有图像上添加遮挡
            h, w = img.shape[:2]
            cv2.rectangle(img, (w//3, h//3), (2*w//3, 2*h//3), (0, 0, 255), -1)
            cv2.imwrite(test_image_path, img)
    
    # 加载图像
    img = load_input_image(test_image_path)
    img_tensor = dreamer.preprocess_image(img)
    
    # 生成相机参数（侧视角，更容易观察遮挡）
    camera_params = {
        'position': [2.0, 0.0, 0.0],  # 侧面位置
        'rotation': [0.0, 90.0, 0.0],  # 旋转90度
        'focal': 1000.0,
        'cx': 256,
        'cy': 256,
        'width': 512,
        'height': 512
    }
    
    # 测试无深度损失的情况
    original_weight = alignment.depth_loss_weight
    alignment.depth_loss_weight = 0.0
    
    # 渲染无深度约束的结果
    rendered_no_depth, _ = gaussian_model.render(camera_params, return_depth=True)
    save_output_image(
        rendered_no_depth,
        os.path.join(output_dir, "no_depth_constraint.jpg")
    )
    
    # 测试有深度损失的情况
    alignment.depth_loss_weight = original_weight
    
    # 渲染有深度约束的结果
    rendered_with_depth, _ = gaussian_model.render(camera_params, return_depth=True)
    save_output_image(
        rendered_with_depth,
        os.path.join(output_dir, "with_depth_constraint.jpg")
    )
    
    # 评估并记录结果
    metrics_no_depth = evaluate_model(rendered_no_depth, img_tensor[:, :512, :512])
    metrics_with_depth = evaluate_model(rendered_with_depth, img_tensor[:, :512, :512])
    
    with open(os.path.join(output_dir, "occlusion_test_results.txt"), 'w') as f:
        f.write("无深度约束的评估结果:\n")
        f.write(f"PSNR: {metrics_no_depth['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics_no_depth['ssim']:.4f}\n\n")
        
        f.write("有深度约束的评估结果:\n")
        f.write(f"PSNR: {metrics_with_depth['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics_with_depth['ssim']:.4f}\n")
    
    print(f"遮挡处理测试完成，结果保存在: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试遮挡处理效果")
    parser.add_argument('--config', type=str, default='configs/lucid_optimized.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    test_occlusion_handling(args.config)
