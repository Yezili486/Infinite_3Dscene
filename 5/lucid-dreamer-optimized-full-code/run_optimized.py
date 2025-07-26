import yaml
import os
import torch
import cv2
from models.dreamer import EnhancedDreamer
from models.alignment import DepthEnhancedAlignment
from models.optimization import ProgressiveTrainer
from models.renderer import GaussianModel
from utils.data_utils import load_input_image, save_output_image, create_input_directory
from utils.metrics import evaluate_model

def main(config_path):
    """主函数：运行优化后的LucidDreamer"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子，保证可复现性
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available() and config['device'] == 'cuda':
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.benchmark = True
    
    # 检查输入目录并创建
    input_path = create_input_directory(config['paths']['input_dir'])
    
    # 初始化各个模块
    print("初始化模型组件...")
    dreamer = EnhancedDreamer(config)
    gaussian_model = GaussianModel(config)
    alignment = DepthEnhancedAlignment(config, gaussian_model)
    
    # 初始化训练器并开始训练
    print("开始训练优化...")
    trainer = ProgressiveTrainer(config, dreamer, alignment, gaussian_model)
    trainer.train()
    
    # 训练完成后，渲染结果
    print("训练完成，渲染结果...")
    camera_params_list = [
        # 正面视角
        {
            'position': [0.0, 0.0, 3.0],
            'rotation': [0.0, 0.0, 0.0],
            'focal': 1000.0,
            'cx': 512,
            'cy': 512,
            'width': 1024,
            'height': 1024
        },
        # 45度视角
        {
            'position': [2.1, 0.0, 2.1],
            'rotation': [0.0, 45.0, 0.0],
            'focal': 1000.0,
            'cx': 512,
            'cy': 512,
            'width': 1024,
            'height': 1024
        },
        # 侧面视角
        {
            'position': [3.0, 0.0, 0.0],
            'rotation': [0.0, 90.0, 0.0],
            'focal': 1000.0,
            'cx': 512,
            'cy': 512,
            'width': 1024,
            'height': 1024
        }
    ]
    
    # 加载输入图像作为参考
    input_img = load_input_image(input_path)
    input_tensor = dreamer.preprocess_image(input_img)
    
    # 渲染多个视角并保存
    for i, cam_params in enumerate(camera_params_list):
        rendered_img, rendered_depth = gaussian_model.render(cam_params, return_depth=True)
        
        # 保存渲染结果
        save_output_image(
            rendered_img,
            os.path.join(config['paths']['output_dir'], f"rendered_view_{i+1}.jpg")
        )
        
        # 保存深度图
        depth_normalized = (rendered_depth - rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min() + 1e-8)
        save_output_image(
            depth_normalized.repeat(3, 1, 1),  # 转为3通道以便显示
            os.path.join(config['paths']['output_dir'], f"depth_map_view_{i+1}.jpg")
        )
        
        # 评估渲染质量
        if i == 0:  # 只评估正面视角
            metrics = evaluate_model(rendered_img, input_tensor)
            print(f"正面视角评估结果: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
            
            # 保存评估结果
            with open(os.path.join(config['paths']['output_dir'], "evaluation_results.txt"), 'w') as f:
                f.write("渲染质量评估结果:\n")
                f.write(f"PSNR (峰值信噪比): {metrics['psnr']:.2f} dB\n")
                f.write(f"SSIM (结构相似性): {metrics['ssim']:.4f}\n")
    
    print(f"所有结果已保存至: {config['paths']['output_dir']}")
    print("程序运行完成!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="运行优化后的LucidDreamer 3D生成模型")
    parser.add_argument('--config', type=str, default='configs/lucid_optimized.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)
