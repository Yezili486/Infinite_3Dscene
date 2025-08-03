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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = DemoConfig(device=args.device)
    os.makedirs(args.output, exist_ok=True)

    processor = DataProcessor(config)
    input_data = processor.load_images(args.input)

    model = ModelWrapper(config)
    results = model.process(input_data)

    processor.save_results(results, args.output)

    visualizer = ResultVisualizer()
    visualizer.generate_report(args.output, args.ref, os.path.join(args.output, "对比报告.html"))

    print(f"完成！结果在 {args.output}")

if __name__ == "__main__":
    main()