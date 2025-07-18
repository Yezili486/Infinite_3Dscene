#!/usr/bin/env python3
"""
Generative Powers of Ten - 编程示例

这个脚本展示如何在不使用Gradio界面的情况下直接使用核心功能
"""

from gradio_powers_of_ten_demo import (
    create_zoom_stack, 
    Pi_image, 
    Pi_noise,
    joint_multi_scale_sampling_demo,
    tensor_to_pil,
    create_artistic_pattern
)
import torch

def example_1_basic_zoom_stack():
    """示例1：创建基础缩放栈"""
    print("\n=== 示例1：创建基础缩放栈 ===")
    
    # 创建缩放栈
    zoom_factors = [1, 2, 4, 8]
    zoom_stack = create_zoom_stack(zoom_factors, H=256, W=256, device="cpu")
    
    print(f"缩放栈信息:")
    print(f"  - 层数: {zoom_stack.N}")
    print(f"  - 尺寸: {zoom_stack.H}x{zoom_stack.W}")
    print(f"  - 缩放因子: {zoom_stack.zoom_factors}")
    
    # 渲染不同层
    for i in range(zoom_stack.N):
        rendered = Pi_image(zoom_stack, i)
        print(f"  - 层 {i}: 缩放{zoom_factors[i]}x, 范围[{rendered.min():.3f}, {rendered.max():.3f}]")
    
    return zoom_stack

def example_2_pattern_generation():
    """示例2：图案生成"""
    print("\n=== 示例2：图案生成 ===")
    
    prompts = [
        "distant galaxy with swirling arms",
        "star system with planets", 
        "alien planet surface",
        "microscopic life forms"
    ]
    
    H, W = 256, 256
    
    for i, prompt in enumerate(prompts):
        pattern = create_artistic_pattern(prompt, H, W, i, len(prompts))
        print(f"生成图案 {i+1}: '{prompt}'")
        print(f"  - 范围: [{pattern.min():.3f}, {pattern.max():.3f}]")
        print(f"  - 形状: {pattern.shape}")
        
        # 保存为图像
        pil_img = tensor_to_pil(pattern)
        filename = f"pattern_{i+1}.png"
        pil_img.save(filename)
        print(f"  - 已保存: {filename}")

def example_3_complete_generation():
    """示例3：完整生成流程"""
    print("\n=== 示例3：完整生成流程 ===")
    
    # 配置
    prompts = [
        "cosmic nebula in deep space",
        "solar system with multiple worlds",
        "rocky planet with atmosphere", 
        "alien forest with strange trees"
    ]
    zoom_factors = [1, 2, 4, 8]
    
    print(f"提示: {prompts}")
    print(f"缩放因子: {zoom_factors}")
    
    # 生成缩放栈
    print("开始生成...")
    result_stack = joint_multi_scale_sampling_demo(
        prompts=prompts,
        zoom_factors=zoom_factors,
        T=10,  # 快速演示
        H=256, W=256
    )
    
    print("生成完成!")
    
    # 保存所有层
    for i in range(result_stack.N):
        rendered = Pi_image(result_stack, i)
        pil_img = tensor_to_pil(rendered)
        filename = f"scale_{i}_zoom_{zoom_factors[i]}x.png"
        pil_img.save(filename)
        print(f"已保存层 {i}: {filename}")
    
    return result_stack

def example_4_animation_frames():
    """示例4：创建动画帧"""
    print("\n=== 示例4：创建动画帧 ===")
    
    # 使用之前生成的缩放栈
    zoom_factors = [1, 2, 4]
    prompts = ["ocean waves", "underwater scene", "coral detail"]
    
    zoom_stack = joint_multi_scale_sampling_demo(
        prompts=prompts,
        zoom_factors=zoom_factors,
        T=5,  # 非常快速
        H=128, W=128
    )
    
    # 创建动画帧
    frames = []
    for i in range(zoom_stack.N):
        rendered = Pi_image(zoom_stack, i)
        pil_img = tensor_to_pil(rendered)
        frames.append(pil_img)
        print(f"创建帧 {i+1}/{zoom_stack.N}")
    
    # 保存为GIF
    if frames:
        output_path = "example_animation.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,
            loop=0
        )
        print(f"动画已保存: {output_path}")
    
    return frames

def example_5_noise_rendering():
    """示例5：噪声渲染演示"""
    print("\n=== 示例5：噪声渲染演示 ===")
    
    zoom_factors = [1, 2, 4]
    zoom_stack = create_zoom_stack(zoom_factors, H=128, W=128, device="cpu")
    
    # 为每层设置不同的内容
    for i in range(zoom_stack.N):
        # 创建测试图案
        test_pattern = torch.zeros((128, 128, 3))
        center = 64
        radius = 30 // (i + 1)
        
        for x in range(128):
            for y in range(128):
                dist = ((x - center)**2 + (y - center)**2)**0.5
                if dist < radius:
                    test_pattern[x, y, i % 3] = 0.5
        
        zoom_stack.set_layer(i, test_pattern * 2 - 1)  # 转换到[-1,1]
    
    # 渲染图像和噪声
    for i in range(zoom_stack.N):
        # 渲染图像
        img = Pi_image(zoom_stack, i)
        img_pil = tensor_to_pil(img)
        img_pil.save(f"noise_test_img_{i}.png")
        
        # 渲染噪声
        noise = Pi_noise(zoom_stack, i)
        # 噪声可视化（归一化到[0,1]）
        noise_vis = (noise - noise.min()) / (noise.max() - noise.min())
        noise_pil = tensor_to_pil(noise_vis * 2 - 1)
        noise_pil.save(f"noise_test_noise_{i}.png")
        
        print(f"层 {i}: 图像范围[{img.min():.3f}, {img.max():.3f}], "
              f"噪声范围[{noise.min():.3f}, {noise.max():.3f}]")

def main():
    """主函数 - 运行所有示例"""
    print("=== Generative Powers of Ten - 编程示例 ===")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 运行示例
        example_1_basic_zoom_stack()
        example_2_pattern_generation()
        result_stack = example_3_complete_generation()
        example_4_animation_frames()
        example_5_noise_rendering()
        
        print(f"\n✅ 所有示例运行完成!")
        print("生成的文件:")
        print("  - pattern_*.png: 不同主题的图案")
        print("  - scale_*.png: 多尺度生成结果")
        print("  - example_animation.gif: 缩放动画")
        print("  - noise_test_*.png: 噪声渲染测试")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")

if __name__ == "__main__":
    main() 