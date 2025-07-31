import argparse
import torch
from models.closeup_gs import CloseupGaussianEnhancer
from models.esrgan import ESRGANEnhancer
from models.zoedepth import ZoeDepthEstimator
from models.gaussian_renderer import render

def main(args):
    # 1. 初始化设备
    device = torch.device(args.device)

    # 2. 加载模型（整合 Closeup GS + LucidDreamer 模块）
    esrgan = ESRGANEnhancer(args.esrgan_weight).to(device)
    zoedepth = ZoeDepthEstimator(args.zoedepth_weight).to(device)
    closeup_enhancer = CloseupGaussianEnhancer().to(device)
    closeup_enhancer.load_state_dict(torch.load(args.closeup_gs_weight, map_location=device))

    # 3. 数据预处理（近景物体图片）
    img = esrgan.enhance(args.input_image)  # ESRGAN 超分（复用 LucidDreamer）
    depth_map = zoedepth.estimate(img)      # ZoeDepth 深度估计（复用）
    point_cloud = generate_point_cloud(img, depth_map)  # 从图+深度生成初始点云

    # 4. Closeup GS 核心：精细化点云
    enhanced_pc = closeup_enhancer(point_cloud.to(device))  # 增强近景细节

    # 5. 3DGS 渲染
    render_results = render(
        enhanced_pc, 
        cameras=args.cameras_path, 
        model_path=args.3dgs_weight,
        output_dir=args.output_dir
    )
    print(f"渲染完成，结果保存至 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 输入输出参数
    parser.add_argument("--input_image", required=True, help="近景物体图片路径")
    parser.add_argument("--cameras_path", required=True, help="相机参数 JSON 路径")
    parser.add_argument("--output_dir", default="./results")
    # 权重路径
    parser.add_argument("--closeup_gs_weight", default="./pretrained/closeup_gs.pth")
    parser.add_argument("--esrgan_weight", default="./pretrained/RealESRGAN_x4plus.pth")
    parser.add_argument("--zoedepth_weight", default="./pretrained/model.safetensors")
    parser.add_argument("--3dgs_weight", default="./pretrained/3dgs_base.pth")
    # 设备参数
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
    