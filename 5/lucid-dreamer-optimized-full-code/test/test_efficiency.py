import yaml
import os
import time
import torch
import numpy as np
from models.dreamer import EnhancedDreamer
from models.alignment import DepthEnhancedAlignment
from models.optimization import ProgressiveTrainer
from models.renderer import GaussianModel
from utils.data_utils import load_input_image

def test_training_efficiency(config_path):
    """测试训练效率"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = os.path.join(config['paths']['output_dir'], 'efficiency_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化组件
    dreamer = EnhancedDreamer(config)
    
    # 记录结果的字典
    results = {}
    
    # 1. 测试原始训练策略（固定高分辨率）
    print("测试原始训练策略...")
    gaussian_model = GaussianModel(config)
    alignment = DepthEnhancedAlignment(config, gaussian_model)
    
    # 修改为固定高分辨率训练
    original_stage1_res = config['model']['progressive_training']['stage1_res']
    original_stage1_epochs = config['model']['progressive_training']['stage1_epochs']
    config['model']['progressive_training']['stage1_res'] = config['model']['progressive_training']['stage2_res']
    config['model']['progressive_training']['stage1_epochs'] = 300  # 总轮数保持一致
    
    trainer = ProgressiveTrainer(config, dreamer, alignment, gaussian_model)
    
    start_time = time.time()
    trainer.train()  # 实际上只进行第一阶段训练
    original_time = time.time() - start_time
    
    # 记录结果
    results['original'] = {
        'time': original_time,
        'epochs': 300,
        'resolution': config['model']['progressive_training']['stage2_res']
    }
    
    # 恢复配置
    config['model']['progressive_training']['stage1_res'] = original_stage1_res
    config['model']['progressive_training']['stage1_epochs'] = original_stage1_epochs
    
    # 2. 测试渐进式训练策略
    print("测试渐进式训练策略...")
    gaussian_model_prog = GaussianModel(config)
    alignment_prog = DepthEnhancedAlignment(config, gaussian_model_prog)
    trainer_prog = ProgressiveTrainer(config, dreamer, alignment_prog, gaussian_model_prog)
    
    start_time = time.time()
    trainer_prog.train()
    progressive_time = time.time() - start_time
    
    # 记录结果
    results['progressive'] = {
        'time': progressive_time,
        'epochs': 300,
        'resolution': config['model']['progressive_training']['stage2_res']
    }
    
    # 计算效率提升
    time_saving = (original_time - progressive_time) / original_time * 100
    
    # 保存结果
    with open(os.path.join(output_dir, "efficiency_results.txt"), 'w') as f:
        f.write("训练效率对比测试结果:\n\n")
        
        f.write("1. 原始训练策略（固定高分辨率）:\n")
        f.write(f"   总耗时: {results['original']['time']:.2f} 秒\n")
        f.write(f"   训练轮数: {results['original']['epochs']}\n")
        f.write(f"   分辨率: {results['original']['resolution']}x{results['original']['resolution']}\n\n")
        
        f.write("2. 渐进式训练策略:\n")
        f.write(f"   总耗时: {results['progressive']['time']:.2f} 秒\n")
        f.write(f"   训练轮数: {results['progressive']['epochs']}\n")
        f.write(f"   分辨率: 先 {original_stage1_res}x{original_stage1_res}, 后 {config['model']['progressive_training']['stage2_res']}x{config['model']['progressive_training']['stage2_res']}\n\n")
        
        f.write(f"效率提升: {time_saving:.2f}%\n")
    
    print(f"训练效率测试完成，结果保存在: {output_dir}")
    print(f"渐进式训练比原始训练快 {time_saving:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试训练效率")
    parser.add_argument('--config', type=str, default='configs/lucid_optimized.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    test_training_efficiency(args.config)
