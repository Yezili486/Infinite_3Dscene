import torch
import numpy as np
from realesrgan import RealESRGANer
from zoedepth.models import ZoeDepth
from gaussian_splatting import GaussianRenderer

class ModelWrapper:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.models = self._load_models()

    def _load_models(self):
        return {
            "esrgan": RealESRGANer(scale=4, model_path=self.config.pretrained["esrgan"], device=self.device),
            "zoe": ZoeDepth.from_pretrained().to(self.device).eval(),
            "closeup_gs": torch.load(self.config.pretrained["closeup_gs"], map_location=self.device).eval(),
            "renderer": GaussianRenderer(device=self.device)
        }

    def process(self, input_data):
        results = []
        with torch.no_grad():
            for item in input_data:
                sr_img, _ = self.models["esrgan"].enhance(item["data"])
                depth = self.models["zoe"].infer(sr_img)
                pc = self._create_point_cloud(sr_img, depth)
                enhanced_pc = self.models["closeup_gs"](pc)
                renders = [self.models["renderer"](enhanced_pc, angle=a, iterations=500) for a in self.config.render_views]
                results.append({"name": item["name"], "renders": renders})
        return results

    def _create_point_cloud(self, img, depth):
        h, w = img.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth / 1000.0
        x3d = (xx - w//2) * z / (w/2)
        y3d = (yy - h//2) * z / (h/2)
        points = np.stack([x3d, y3d, z], axis=-1).reshape(-1, 3)
        colors = img.reshape(-1, 3) / 255.0
        idx = np.random.choice(len(points), self.config.point_cloud_density, replace=False)
        return {
            "points": torch.tensor(points[idx], dtype=torch.float32).to(self.device),
            "colors": torch.tensor(colors[idx], dtype=torch.float32).to(self.device)
        }