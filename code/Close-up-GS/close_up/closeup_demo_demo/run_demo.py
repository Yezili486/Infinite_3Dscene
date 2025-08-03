import os
import argparse
from config import DemoConfig
from data_processor import DataProcessor
from model_wrapper import ModelWrapper
from result_visualizer import ResultVisualizer

def main():
    parser = argparse.ArgumentParser(description="Closeup GS Demo")
    parser.add_argument("--input", default="./input", help="输入图片目录")
    parser.add_argument("--output", default="./output", help="结果输出目录")
    parser.add_argument("--ref", default="./reference", help="官方参考图目录")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Close-up GS Demo 启动")
    print("=" * 60)

    # 初始化配置
    config = DemoConfig(device=args.device)
    print(f"使用设备: {config.device}")
    print(f"点云密度: {config.point_cloud_density}")
    print(f"渲染迭代: {config.render_iterations}")
    print(f"渲染视角: {config.render_views}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"输出目录: {args.output}")

    # 数据处理器
    processor = DataProcessor(config)
    input_data = processor.load_images(args.input)
    print(f"加载了 {len(input_data)} 张图像")

    # 模型包装器
    model = ModelWrapper(config)
    results = model.process(input_data)

    # 保存结果
    processor.save_results(results, args.output)
    print(f"结果已保存到: {args.output}")

    # 生成可视化报告
    visualizer = ResultVisualizer()
    report_path = os.path.join(args.output, "对比报告.html")
    visualizer.generate_report(args.output, args.ref, report_path)

    print("\n" + "=" * 60)
    print("🎉 Demo 运行完成!")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print("生成的文件:")
    
    if os.path.exists(args.output):
        for file in os.listdir(args.output):
            file_path = os.path.join(args.output, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size:,} bytes)")
    
    print(f"\n📊 统计信息:")
    print(f"  • 处理图像数: {len(input_data)}")
    print(f"  • 生成结果数: {len(results)}")
    print(f"  • 总渲染数: {len(results) * len(config.render_views)}")
    
    print(f"\n🔗 查看结果:")
    print(f"  • 对比报告: {report_path}")
    print(f"  • 输出目录: {os.path.abspath(args.output)}")
    
    print("\n✨ Demo 运行成功!")

if __name__ == "__main__":
    main() 