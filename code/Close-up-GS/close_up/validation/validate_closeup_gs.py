import os
import cv2
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class CloseupGSValidator:
    def __init__(self, reference_dir, test_dir, output_report="validation_report.json"):
        """
        初始化验证器
        :param reference_dir: 官方参考结果目录
        :param test_dir: 待验证的测试结果目录
        :param output_report: 验证报告输出路径
        """
        self.reference_dir = reference_dir
        self.test_dir = test_dir
        self.output_report = output_report
        self.results = {}
        
        # 检查目录是否存在
        assert os.path.exists(reference_dir), f"参考目录不存在: {reference_dir}"
        assert os.path.exists(test_dir), f"测试目录不存在: {test_dir}"

    def calculate_psnr(self, img1_path, img2_path):
        """计算两张图片的PSNR值"""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # 确保图片尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        mse = np.mean((img1 - img2) **2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255** 2 / mse)

    def calculate_ssim(self, img1_path, img2_path):
        """计算两张图片的SSIM值"""
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图片尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        C1 = (0.01 * 255) **2
        C2 = (0.03 * 255)** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 **2
        mu2_sq = mu2** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 **2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def check_3d_consistency(self, render_dir):
        """检查3D渲染的视角一致性"""
        render_files = sorted([f for f in os.listdir(render_dir) if f.endswith(('.png', '.jpg'))])
        if len(render_files) < 3:
            return {"status": "warning", "message": "渲染视角不足3个，无法验证一致性"}
        
        # 检查相邻视角的差异是否平滑
        diffs = []
        for i in range(len(render_files) - 1):
            img1 = cv2.imread(os.path.join(render_dir, render_files[i]))
            img2 = cv2.imread(os.path.join(render_dir, render_files[i+1]))
            img1 = cv2.resize(img1, (512, 512))
            img2 = cv2.resize(img2, (512, 512))
            diff = np.mean(np.abs(img1 - img2))
            diffs.append(diff)
        
        avg_diff = np.mean(diffs)
        if avg_diff < 15:  # 阈值可根据实际情况调整
            return {
                "status": "pass", 
                "average_difference": avg_diff,
                "message": "视角转换平滑，3D一致性良好"
            }
        else:
            return {
                "status": "fail", 
                "average_difference": avg_diff,
                "message": "视角转换差异过大，3D一致性较差"
            }

    def validate_detail_preservation(self, reference_detail_path, test_detail_path):
        """验证近景细节保留效果"""
        # 读取细节区域（假设已标注ROI）
        ref_img = cv2.imread(reference_detail_path)
        test_img = cv2.imread(test_detail_path)
        
        if ref_img is None or test_img is None:
            return {"status": "error", "message": "细节区域图片不存在"}
            
        # 转为灰度图并计算梯度（边缘检测）
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        ref_edges = cv2.Canny(ref_gray, 100, 200)
        test_edges = cv2.Canny(test_gray, 100, 200)
        
        # 计算边缘相似度
        edge_similarity = np.mean(ref_edges == test_edges)
        
        if edge_similarity > 0.75:  # 75%以上边缘匹配
            return {
                "status": "pass", 
                "edge_similarity": edge_similarity,
                "message": "近景细节保留良好"
            }
        else:
            return {
                "status": "fail", 
                "edge_similarity": edge_similarity,
                "message": "近景细节丢失较多"
            }

    def generate_visual_comparison(self):
        """生成视觉对比图"""
        os.makedirs("visual_comparison", exist_ok=True)
        
        # 对比关键结果
        for img_name in ["enhanced_image.jpg", "depth_map.png", "render_000.jpg"]:
            ref_path = os.path.join(self.reference_dir, img_name)
            test_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(ref_path) or not os.path.exists(test_path):
                continue
                
            ref_img = Image.open(ref_path)
            test_img = Image.open(test_path)
            
            # 调整尺寸一致
            test_img = test_img.resize(ref_img.size)
            
            # 创建对比图
            combined = Image.new('RGB', (ref_img.width * 2, ref_img.height))
            combined.paste(ref_img, (0, 0))
            combined.paste(test_img, (ref_img.width, 0))
            
            # 添加标签
            plt.figure(figsize=(12, 6))
            plt.imshow(combined)
            plt.text(ref_img.width/2, 30, "官方参考结果", fontsize=12, color='red', ha='center')
            plt.text(ref_img.width*3/2, 30, "复现结果", fontsize=12, color='green', ha='center')
            plt.axis('off')
            plt.savefig(f"visual_comparison/compare_{img_name}", bbox_inches='tight')
            plt.close()

    def run_validation(self):
        """运行完整验证流程"""
        print("开始验证Closeup GS复现结果...")
        
        # 1. 渲染结果质量评估
        render_psnr = self.calculate_psnr(
            os.path.join(self.reference_dir, "render_000.jpg"),
            os.path.join(self.test_dir, "render_000.jpg")
        )
        
        render_ssim = self.calculate_ssim(
            os.path.join(self.reference_dir, "render_000.jpg"),
            os.path.join(self.test_dir, "render_000.jpg")
        )
        
        # 2. 3D一致性检查
        consistency_result = self.check_3d_consistency(os.path.join(self.test_dir, "3d_render"))
        
        # 3. 细节保留验证
        detail_result = self.validate_detail_preservation(
            os.path.join(self.reference_dir, "detail_region.jpg"),
            os.path.join(self.test_dir, "detail_region.jpg")
        )
        
        # 4. 综合评估
        self.results = {
            "image_quality": {
                "psnr": round(render_psnr, 2),
                "psnr_pass": render_psnr >= 25.0,  # 官方标准通常>25dB
                "ssim": round(render_ssim, 3),
                "ssim_pass": render_ssim >= 0.85   # 官方标准通常>0.85
            },
            "3d_consistency": consistency_result,
            "detail_preservation": detail_result,
            "overall_pass": all([
                render_psnr >= 25.0,
                render_ssim >= 0.85,
                consistency_result["status"] == "pass",
                detail_result["status"] == "pass"
            ])
        }
        
        # 生成视觉对比图
        self.generate_visual_comparison()
        
        # 保存验证报告
        with open(self.output_report, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"验证完成，报告已保存至 {self.output_report}")
        print(f"综合验证结果: {'通过' if self.results['overall_pass'] else '未通过'}")

if __name__ == "__main__":
    # 示例用法
    validator = CloseupGSValidator(
        reference_dir="./official_reference_results",  # 官方参考结果目录
        test_dir="./your_test_results",                # 你的复现结果目录
        output_report="closeup_gs_validation_report.json"
    )
    validator.run_validation()
    