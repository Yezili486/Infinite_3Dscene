import os
import cv2
import numpy as np
from PIL import Image

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_images(self, input_dir):
        images = []
        for f in os.listdir(input_dir):
            if f.lower().endswith((".jpg", ".png")):
                img = Image.open(os.path.join(input_dir, f)).convert("RGB")
                images.append({"name": f.split(".")[0], "data": np.array(img)})
        return images

    def save_results(self, results, output_dir):
        for res in results:
            for angle, img in zip(self.config.render_views, res["renders"]):
                cv2.imwrite(
                    os.path.join(output_dir, f"{res['name']}_view{angle}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )