import os
import cv2
import numpy as np
from PIL import Image

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_images(self, input_dir):
        images = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith((".jpg", ".png")):
                    img = Image.open(os.path.join(input_dir, f)).convert("RGB")
                    images.append({"name": f.split(".")[0], "data": np.array(img)})
        else:
            # 创建示例图像
            print(f"输入目录 {input_dir} 不存在，创建示例图像...")
            os.makedirs(input_dir, exist_ok=True)
            sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            sample_path = os.path.join(input_dir, "sample.jpg")
            Image.fromarray(sample_image).save(sample_path)
            images.append({"name": "sample", "data": sample_image})
        return images

    def save_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for res in results:
            for angle, img in zip(self.config.render_views, res["renders"]):
                cv2.imwrite(
                    os.path.join(output_dir, f"{res['name']}_view{angle}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ) 