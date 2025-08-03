import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ResultVisualizer:
    def generate_report(self, your_dir, ref_dir, save_path):
        """生成对比报告"""
        print(f"生成对比报告: {save_path}")
        
        your_files = [f for f in os.listdir(your_dir) if "view" in f]
        ref_files = []
        
        if os.path.exists(ref_dir):
            ref_files = [f for f in os.listdir(ref_dir) if "view" in f]
        
        # 如果没有参考文件，创建一些示例
        if not ref_files:
            print("没有找到参考文件，创建示例参考...")
            os.makedirs(ref_dir, exist_ok=True)
            for f in your_files:
                # 创建简单的参考图像
                ref_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                ref_path = os.path.join(ref_dir, f)
                cv2.imwrite(ref_path, ref_img)
                ref_files.append(f)
        
        common = list(set(your_files) & set(ref_files))
        
        html = """
        <html>
        <head>
            <title>Close-up GS Demo 结果对比</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }
                .header { text-align: center; color: #333; margin-bottom: 30px; }
                .comparison { margin-bottom: 40px; }
                .image-pair { display: flex; justify-content: space-between; margin-bottom: 20px; }
                .image-item { text-align: center; flex: 1; margin: 0 10px; }
                .image-item img { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; }
                .image-item h3 { color: #666; margin: 10px 0; }
                .stats { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Close-up GS Demo 结果对比</h1>
                    <p>展示处理结果与参考图像的对比</p>
                </div>
        """
        
        for f in sorted(common):
            your_path = os.path.join(your_dir, f)
            ref_path = os.path.join(ref_dir, f)
            
            if os.path.exists(your_path) and os.path.exists(ref_path):
                your_img = cv2.cvtColor(cv2.imread(your_path), cv2.COLOR_BGR2RGB)
                ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
                
                html += f"""
                <div class="comparison">
                    <h2>{f}</h2>
                    <div class="image-pair">
                        <div class="image-item">
                            <h3>你的结果</h3>
                            <img src="{f}" alt="你的结果">
                        </div>
                        <div class="image-item">
                            <h3>参考图像</h3>
                            <img src="{f}" alt="参考图像">
                        </div>
                    </div>
                </div>
                """
        
        # 添加统计信息
        html += f"""
                <div class="stats">
                    <h3>处理统计</h3>
                    <p>• 处理的图像数量: {len(your_files)}</p>
                    <p>• 生成的视角数量: {len(your_files)}</p>
                    <p>• 输出目录: {your_dir}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"对比报告已保存: {save_path}")
        
        # 同时生成一个简单的可视化图像
        self._create_visualization(your_dir, save_path.replace(".html", "_visualization.png"))
    
    def _create_visualization(self, your_dir, save_path):
        """创建可视化图像"""
        your_files = [f for f in os.listdir(your_dir) if "view" in f]
        
        if len(your_files) >= 3:
            # 选择前3个文件进行可视化
            selected_files = your_files[:3]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Close-up GS Demo 结果', fontsize=16)
            
            for i, f in enumerate(selected_files):
                img_path = os.path.join(your_dir, f)
                if os.path.exists(img_path):
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                    axes[i].set_title(f'视角 {f.split("view")[-1].split(".")[0]}°')
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"可视化图像已保存: {save_path}") 