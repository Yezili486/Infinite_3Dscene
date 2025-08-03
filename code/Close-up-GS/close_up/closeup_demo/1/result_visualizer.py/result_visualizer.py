import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_html5agg import FigureCanvasHtml5Agg

class ResultVisualizer:
    def generate_report(self, your_dir, ref_dir, save_path):
        your_files = [f for f in os.listdir(your_dir) if "view" in f]
        ref_files = [f for f in os.listdir(ref_dir) if "view" in f]
        common = list(set(your_files) & set(ref_files))
        html = "<html><body><h1>结果对比</h1>"
        for f in sorted(common):
            your_img = cv2.cvtColor(cv2.imread(os.path.join(your_dir, f)), cv2.COLOR_BGR2RGB)
            ref_img = cv2.cvtColor(cv2.imread(os.path.join(ref_dir, f)), cv2.COLOR_BGR2RGB)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(your_img); ax1.set_title("你的结果")
            ax2.imshow(ref_img); ax2.set_title("官方参考")
            plt.suptitle(f)
            canvas = FigureCanvasHtml5Agg(fig)
            canvas.draw()
            html += f"<h3>{f}</h3>" + canvas.get_snapshot()
            plt.close(fig)
        with open(save_path, "w") as f:
            f.write(html)