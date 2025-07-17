#!/usr/bin/env python3
"""
Generative Powers of Ten - 使用示例

这个文件展示了如何使用完整的"Generative Powers of Ten"实现
包括标准多尺度生成和基于照片的缩放功能

作者: 基于论文 "Generative Powers of Ten" 实现
"""

import torch
from generative_powers_of_ten import *

def example_1_basic_multi_scale_generation():
    """示例1: 基础多尺度生成"""
    print("=== 示例1: 基础多尺度生成 ===")
    
    # 定义多尺度提示
    prompts = [
        "vast cosmic background with nebulae",
        "spiral galaxy with bright stars", 
        "solar system with planets orbiting",
        "detailed surface of an alien planet"
    ]
    
    # 定义缩放因子（2的幂序列）
    zoom_factors = [1, 2, 4, 8]
    
    print(f"提示: {prompts}")
    print(f"缩放因子: {zoom_factors}")
    
    # 运行简化版本（无需完整Stable Diffusion）
    result_stack = joint_multi_scale_sampling_simple(
        prompts=prompts,
        zoom_factors=zoom_factors,
        T=20,  # 扩散步数
        H=256, W=256  # 图像尺寸
    )
    
    # 保存结果
    for i, zoom in enumerate(zoom_factors):
        filename = f"cosmic_scale_{i}_zoom_{zoom}x.png"
        result_stack.save_layer_as_image(i, filename)
        print(f"已保存: {filename}")
    
    return result_stack


def example_2_photo_based_generation():
    """示例2: 基于照片的生成"""
    print("\n=== 示例2: 基于照片的生成 ===")
    
    # 创建输入图像（可以是任何图像）
    H, W = 256, 256
    input_image = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    
    # 创建一个风景图案作为示例
    # 天空渐变
    for i in range(H//2):
        input_image[i, :, 2] = 0.8 * (1 - i/(H//2))  # 蓝色天空
    
    # 地面
    input_image[H//2:, :, 1] = 0.6  # 绿色地面
    
    # 太阳
    center_x, center_y = H//4, W//4
    for i in range(H):
        for j in range(W):
            dist = ((i - center_x)**2 + (j - center_y)**2)**0.5
            if dist < 20:
                input_image[i, j, 0] = 0.9  # 黄色太阳
                input_image[i, j, 1] = 0.9
    
    print(f"创建输入风景图像: {input_image.shape}")
    
    # 定义增强提示
    prompts = [
        "enhance the landscape with artistic style",
        "add detailed textures and atmosphere", 
        "create intricate surface details"
    ]
    zoom_factors = [1, 2, 4]
    
    # 运行基于照片的生成
    result_stack = joint_multi_scale_sampling_with_photo_simple(
        prompts=prompts,
        zoom_factors=zoom_factors,
        input_image=input_image,
        T=15,
        optimize_steps=3,  # 每步优化次数
        optimize_lr=0.05,  # 优化学习率
        H=H, W=W
    )
    
    # 保存输入和结果
    import numpy as np
    from PIL import Image
    
    # 保存输入图像
    input_pil = (torch.clamp(input_image, 0, 1).cpu().numpy() * 255).astype('uint8')
    Image.fromarray(input_pil).save("input_landscape.png")
    print("已保存输入图像: input_landscape.png")
    
    # 保存生成的各层
    for i, zoom in enumerate(zoom_factors):
        filename = f"landscape_enhanced_scale_{i}_zoom_{zoom}x.png"
        result_stack.save_layer_as_image(i, filename)
        print(f"已保存: {filename}")
    
    return result_stack


def example_3_custom_zoom_stack():
    """示例3: 自定义缩放栈操作"""
    print("\n=== 示例3: 自定义缩放栈操作 ===")
    
    # 创建自定义缩放栈
    zoom_factors = [1, 2, 4, 8, 16]  # 5层缩放
    zoom_stack = create_zoom_stack(zoom_factors, H=128, W=128, device=device)
    
    zoom_stack.print_info()
    
    # 手动设置每层的内容
    for i, zoom in enumerate(zoom_factors):
        # 创建不同颜色的图案
        layer = torch.zeros((128, 128, 3), device=device, dtype=torch.float32)
        
        # 根据缩放级别创建不同的图案
        size = 64 // zoom
        start = 64 - size // 2
        end = start + size
        
        # 不同层使用不同颜色
        colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), 
                 (0.8, 0.8, 0.2), (0.8, 0.2, 0.8)]
        
        r, g, b = colors[i % len(colors)]
        layer[start:end, start:end, 0] = r
        layer[start:end, start:end, 1] = g 
        layer[start:end, start:end, 2] = b
        
        zoom_stack.set_layer(i, layer)
    
    # 测试渲染函数
    print("\n测试渲染函数:")
    for i in range(len(zoom_factors)):
        rendered_img = Pi_image(zoom_stack, i)
        rendered_noise = Pi_noise(zoom_stack, i)
        
        print(f"层 {i}: 图像范围=[{rendered_img.min():.3f}, {rendered_img.max():.3f}], "
              f"噪声分布=μ{rendered_noise.mean():.3f} σ{rendered_noise.std():.3f}")
        
        # 保存渲染结果
        filename = f"custom_rendered_layer_{i}.png"
        zoom_stack.save_layer_as_image(i, filename)
    
    return zoom_stack


def example_4_advanced_features():
    """示例4: 高级功能演示"""
    print("\n=== 示例4: 高级功能演示 ===")
    
    # 演示DDPM更新步骤
    print("1. DDPM更新步骤测试:")
    test_image = torch.randn((64, 64, 3), device=device) * 0.5
    test_noise = torch.randn((64, 64, 3), device=device)
    
    for t in [100, 50, 10, 1]:
        updated = ddpm_update_simple(test_image, test_image, test_noise, t)
        print(f"   t={t}: 更新成功, 范围=[{updated.min():.3f}, {updated.max():.3f}]")
    
    # 演示多分辨率融合
    print("\n2. 多分辨率融合测试:")
    zoom_factors = [1, 2, 4]
    zoom_stack = create_zoom_stack(zoom_factors, H=64, W=64, device=device)
    
    # 创建测试预测
    predictions = []
    for i in range(len(zoom_factors)):
        pred = torch.randn((64, 64, 3), device=device) * 0.3
        predictions.append(pred)
    
    blended_stack = multi_resolution_blending_simple(predictions, zoom_stack)
    print("   多分辨率融合完成")
    blended_stack.print_info()


def main():
    """主函数：运行所有示例"""
    print("🎯 Generative Powers of Ten - 使用示例")
    print("========================================")
    
    # 设置设备
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 运行所有示例
        example_1_basic_multi_scale_generation()
        example_2_photo_based_generation() 
        example_3_custom_zoom_stack()
        example_4_advanced_features()
        
        print("\n🎉 所有示例运行完成!")
        print("\n查看生成的图像文件以查看结果。")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        print("请确保已正确安装所有依赖并有足够的GPU内存。")


if __name__ == "__main__":
    main() 