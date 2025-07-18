#!/usr/bin/env python3
"""
缩放视频生成示例

展示如何使用 'Generative Powers of Ten' 实现生成缩放视频。
包含多种视频生成模式和完整的使用示例。

依赖:
pip install torch torchvision opencv-python numpy

使用方法:
python zoom_video_example.py
"""

import torch
import numpy as np
from generative_powers_of_ten import (
    create_zoom_stack, Pi_image, Pi_noise,
    render_zoom_video, render_smooth_zoom_video, render_zoom_video_with_effects,
    joint_multi_scale_sampling_simple, joint_multi_scale_sampling_with_photo_simple
)

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

def create_sample_zoom_stack(zoom_factors=[1, 2, 4, 8], H=512, W=512):
    """创建一个示例缩放栈，包含有趣的图案"""
    print(f"\n=== 创建示例缩放栈 ===")
    print(f"缩放因子: {zoom_factors}")
    print(f"分辨率: {H}x{W}")
    
    zoom_stack = create_zoom_stack(zoom_factors, H, W, device)
    
    # 为每一层创建不同的艺术图案
    for i, zoom_factor in enumerate(zoom_factors):
        print(f"生成第 {i+1} 层图案 (缩放因子 {zoom_factor}x)...")
        
        # 创建基础画布
        img = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
        
        # 创建分形螺旋图案
        center_x, center_y = H // 2, W // 2
        
        for x in range(H):
            for y in range(W):
                # 距离中心的距离
                dx, dy = x - center_x, y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # 角度
                angle = np.arctan2(dy, dx)
                
                # 分形参数
                freq = zoom_factor * 2
                spiral_freq = 0.02 * zoom_factor
                
                # 螺旋波纹
                spiral_value = np.sin(dist * spiral_freq + angle * freq)
                
                # 径向波纹
                radial_value = np.sin(dist * 0.05 * zoom_factor)
                
                # 组合模式
                combined = 0.5 * spiral_value + 0.3 * radial_value
                intensity = 0.5 * (1 + combined)
                
                # 根据层数使用不同的颜色主题
                if i == 0:  # 宇宙主题 - 深蓝紫色
                    img[x, y, 0] = intensity * 0.3  # 红
                    img[x, y, 1] = intensity * 0.1  # 绿
                    img[x, y, 2] = intensity * 0.8  # 蓝
                elif i == 1:  # 星云主题 - 紫红色
                    img[x, y, 0] = intensity * 0.8  # 红
                    img[x, y, 1] = intensity * 0.2  # 绿
                    img[x, y, 2] = intensity * 0.6  # 蓝
                elif i == 2:  # 恒星主题 - 橙黄色
                    img[x, y, 0] = intensity * 1.0  # 红
                    img[x, y, 1] = intensity * 0.7  # 绿
                    img[x, y, 2] = intensity * 0.2  # 蓝
                else:  # 表面主题 - 绿色
                    img[x, y, 0] = intensity * 0.2  # 红
                    img[x, y, 1] = intensity * 0.8  # 绿
                    img[x, y, 2] = intensity * 0.3  # 蓝
                
                # 添加一些噪声增加细节
                noise_level = 0.05
                img[x, y, :] += torch.randn(3, device=device) * noise_level
        
        # 归一化到 [-1, 1] 范围
        img = torch.clamp(img * 2.0 - 1.0, -1, 1)
        zoom_stack.set_layer(i, img)
    
    print(f"✅ 示例缩放栈创建完成!")
    zoom_stack.print_info()
    return zoom_stack

def create_fractal_zoom_stack(zoom_factors=[1, 2, 4, 8], H=256, W=256):
    """创建分形缩放栈"""
    print(f"\n=== 创建分形缩放栈 ===")
    
    zoom_stack = create_zoom_stack(zoom_factors, H, W, device)
    
    # Mandelbrot集合参数
    max_iter = 100
    
    for i, zoom_factor in enumerate(zoom_factors):
        print(f"生成分形第 {i+1} 层...")
        
        img = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
        
        # 根据缩放因子调整视野
        zoom_level = zoom_factor * 0.5
        x_min, x_max = -2.0/zoom_level, 1.0/zoom_level
        y_min, y_max = -1.5/zoom_level, 1.5/zoom_level
        
        # 中心偏移（创造有趣的区域）
        center_x = -0.5 + 0.1 * i
        center_y = 0.0 + 0.1 * i
        
        for py in range(H):
            for px in range(W):
                # 映射到复平面
                x = x_min + (x_max - x_min) * px / W + center_x
                y = y_min + (y_max - y_min) * py / H + center_y
                c = complex(x, y)
                
                # Mandelbrot迭代
                z = 0
                for n in range(max_iter):
                    if abs(z) > 2:
                        break
                    z = z*z + c
                
                # 着色
                if n == max_iter:
                    # 在集合内 - 黑色
                    color_val = 0.0
                else:
                    # 在集合外 - 根据迭代次数着色
                    color_val = n / max_iter
                
                # 根据层数使用不同颜色
                if i % 3 == 0:
                    img[py, px, 0] = color_val * 0.8
                    img[py, px, 1] = color_val * 0.2
                    img[py, px, 2] = color_val * 0.9
                elif i % 3 == 1:
                    img[py, px, 0] = color_val * 0.9
                    img[py, px, 1] = color_val * 0.8
                    img[py, px, 2] = color_val * 0.2
                else:
                    img[py, px, 0] = color_val * 0.2
                    img[py, px, 1] = color_val * 0.9
                    img[py, px, 2] = color_val * 0.8
        
        # 归一化到 [-1, 1]
        img = img * 2.0 - 1.0
        zoom_stack.set_layer(i, img)
    
    return zoom_stack

def demo_basic_video_generation():
    """演示基础视频生成"""
    print(f"\n{'='*50}")
    print(f"演示 1: 基础缩放视频生成")
    print(f"{'='*50}")
    
    # 创建示例缩放栈
    zoom_stack = create_sample_zoom_stack([1, 2, 4, 8], H=256, W=256)
    
    # 生成基础缩放视频
    try:
        print(f"\n--- 生成基础缩放视频 ---")
        video_path = render_zoom_video(
            zoom_stack=zoom_stack,
            output_path="demo_basic_zoom.mp4",
            fps=24,
            duration_per_scale=2.0,
            smooth_transitions=True,
            zoom_speed="constant"
        )
        print(f"✅ 基础缩放视频已保存: {video_path}")
        
    except Exception as e:
        print(f"❌ 基础视频生成失败: {e}")

def demo_smooth_video_generation():
    """演示平滑视频生成"""
    print(f"\n{'='*50}")
    print(f"演示 2: 平滑连续缩放视频")
    print(f"{'='*50}")
    
    # 创建分形缩放栈
    zoom_stack = create_fractal_zoom_stack([1, 2, 4, 8, 16], H=256, W=256)
    
    try:
        print(f"\n--- 生成平滑连续缩放视频 ---")
        video_path = render_smooth_zoom_video(
            zoom_stack=zoom_stack,
            output_path="demo_smooth_zoom.mp4",
            fps=30,
            total_duration=8.0,
            start_scale=0,
            end_scale=4
        )
        print(f"✅ 平滑缩放视频已保存: {video_path}")
        
    except Exception as e:
        print(f"❌ 平滑视频生成失败: {e}")

def demo_effects_video_generation():
    """演示特效视频生成"""
    print(f"\n{'='*50}")
    print(f"演示 3: 特效缩放视频")
    print(f"{'='*50}")
    
    # 创建示例缩放栈
    zoom_stack = create_sample_zoom_stack([1, 2, 4, 8], H=256, W=256)
    
    try:
        print(f"\n--- 生成特效缩放视频 ---")
        video_path = render_zoom_video_with_effects(
            zoom_stack=zoom_stack,
            output_path="demo_effects_zoom.mp4",
            fps=24,
            duration_per_scale=3.0,
            add_fade=True,
            add_zoom_burst=True,
            add_text_overlay=True
        )
        print(f"✅ 特效缩放视频已保存: {video_path}")
        
    except Exception as e:
        print(f"❌ 特效视频生成失败: {e}")

def demo_ai_generated_video():
    """演示AI生成内容的视频"""
    print(f"\n{'='*50}")
    print(f"演示 4: AI生成内容缩放视频")
    print(f"{'='*50}")
    
    # 使用AI生成的缩放栈
    prompts = [
        "cosmic void with distant galaxies",
        "spiral galaxy with bright stars", 
        "solar system with planets",
        "Earth surface with continents"
    ]
    zoom_factors = [1, 2, 4, 8]
    
    try:
        print(f"--- 生成AI缩放栈 ---")
        print(f"提示: {prompts}")
        
        ai_zoom_stack = joint_multi_scale_sampling_simple(
            prompts=prompts,
            zoom_factors=zoom_factors,
            T=10,  # 快速测试
            H=256, W=256
        )
        
        if ai_zoom_stack:
            print(f"--- 从AI生成内容创建视频 ---")
            video_path = render_zoom_video(
                zoom_stack=ai_zoom_stack,
                output_path="demo_ai_zoom.mp4", 
                fps=24,
                duration_per_scale=2.5,
                smooth_transitions=True,
                zoom_speed="accelerating"
            )
            print(f"✅ AI生成缩放视频已保存: {video_path}")
        else:
            print(f"❌ AI缩放栈生成失败")
            
    except Exception as e:
        print(f"❌ AI生成视频失败: {e}")

def demo_photo_based_video():
    """演示基于照片的视频生成"""
    print(f"\n{'='*50}")
    print(f"演示 5: 基于照片的缩放视频")
    print(f"{'='*50}")
    
    # 创建示例输入照片
    H, W = 256, 256
    input_photo = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    
    # 创建抽象艺术图案作为"照片"
    center_x, center_y = H // 2, W // 2
    for x in range(H):
        for y in range(W):
            dx, dy = x - center_x, y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            
            # 创建彩色涡旋图案
            r_val = 0.5 * (1 + np.sin(angle * 3 + dist * 0.05))
            g_val = 0.5 * (1 + np.cos(angle * 5 - dist * 0.03))
            b_val = 0.5 * (1 + np.sin(dist * 0.08 + angle * 2))
            
            input_photo[x, y, 0] = r_val * 0.8
            input_photo[x, y, 1] = g_val * 0.6  
            input_photo[x, y, 2] = b_val * 0.9
    
    # 归一化到 [-1, 1]
    input_photo = input_photo * 2.0 - 1.0
    
    try:
        print(f"--- 基于照片生成缩放栈 ---")
        prompts = ["enhance the spiral", "add cosmic details", "stylize the pattern"]
        zoom_factors = [1, 2, 4]
        
        photo_zoom_stack = joint_multi_scale_sampling_with_photo_simple(
            prompts=prompts,
            zoom_factors=zoom_factors,
            input_image=input_photo,
            T=8,
            optimize_steps=3,
            optimize_lr=0.05,
            H=H, W=W
        )
        
        if photo_zoom_stack:
            print(f"--- 从照片约束内容创建视频 ---")
            video_path = render_smooth_zoom_video(
                zoom_stack=photo_zoom_stack,
                output_path="demo_photo_zoom.mp4",
                fps=30,
                total_duration=6.0
            )
            print(f"✅ 基于照片的缩放视频已保存: {video_path}")
        else:
            print(f"❌ 照片缩放栈生成失败")
            
    except Exception as e:
        print(f"❌ 基于照片的视频生成失败: {e}")

def main():
    """主函数 - 运行所有演示"""
    print(f"🎬 缩放视频生成演示")
    print(f"基于 'Generative Powers of Ten' 论文实现")
    print(f"设备: {device}")
    
    # 检查依赖
    try:
        import cv2
        print(f"✅ OpenCV 版本: {cv2.__version__}")
    except ImportError:
        print(f"❌ 缺少 OpenCV，请安装: pip install opencv-python")
        return
    
    # 运行演示
    demo_basic_video_generation()
    demo_smooth_video_generation()
    demo_effects_video_generation()
    demo_ai_generated_video()
    demo_photo_based_video()
    
    print(f"\n{'='*50}")
    print(f"🎉 所有视频演示完成!")
    print(f"{'='*50}")
    print(f"生成的视频文件:")
    print(f"  📹 demo_basic_zoom.mp4 - 基础缩放视频")
    print(f"  📹 demo_smooth_zoom.mp4 - 平滑连续缩放")
    print(f"  📹 demo_effects_zoom.mp4 - 特效缩放视频")
    print(f"  📹 demo_ai_zoom.mp4 - AI生成内容缩放")
    print(f"  📹 demo_photo_zoom.mp4 - 基于照片的缩放")
    
    print(f"\n🔧 高级使用提示:")
    print(f"1. 调整 fps 参数改变视频流畅度")
    print(f"2. 调整 duration_per_scale 控制每层停留时间")
    print(f"3. 使用 zoom_speed='accelerating'/'decelerating' 改变缩放速度")
    print(f"4. 启用 smooth_transitions 获得更平滑的过渡")
    print(f"5. 使用更高分辨率 (512x512) 获得更好质量")
    
    print(f"\n🚀 下一步:")
    print(f"- 尝试不同的提示组合")
    print(f"- 实验不同的缩放因子序列") 
    print(f"- 添加自定义特效")
    print(f"- 集成到更大的视频制作流程中")

if __name__ == "__main__":
    main() 