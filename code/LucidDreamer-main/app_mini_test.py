# Test version of app_mini.py - skipping downloads and compilation
import os
import glob
import platform
import pathlib
import gradio as gr

root = pathlib.Path(__file__).parent
example_root = os.path.join(root, 'examples')

# Skip downloading and compilation for testing
print("Skipping downloads and compilation for testing...")

# Create a dummy LucidDreamer class for testing
class DummyLucidDreamer:
    def __init__(self, save_dir='./'):
        self.save_dir = save_dir
        print("Dummy LucidDreamer initialized")
    
    def run(self, *args):
        print("Dummy run function called")
        return None, None, None
    
    def create(self, *args):
        print("Dummy create function called")
        return None
    
    def render_video(self, *args):
        print("Dummy render_video function called")
        return None, None

ld = DummyLucidDreamer(save_dir='./')

css = """
#run-button {
  background: coral;
  color: white;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div>
            <h1>LucidDreamer Test Interface</h1>
            <h5>Testing basic Gradio interface functionality</h5>
        </div>
        </div>
        """
    )

    with gr.Row():
        result_gallery = gr.Video(label='RGB Video', show_label=True, autoplay=True, format='mp4')
        result_depth = gr.Video(label='Depth Video', show_label=True, autoplay=True, format='mp4')
        result_ply_file = gr.File(label='Gaussian splatting PLY', show_label=True)

    with gr.Row():
        input_image = gr.Image(
            label='Image prompt',
            sources='upload',
            type='pil',
        )

        with gr.Column():
            prompt = gr.Textbox(
                label='Text prompt',
                value='A cozy livingroom',
            )
            run_button = gr.Button(value='Test Run', elem_id='run-button')

    run_button.click(fn=ld.run, inputs=[input_image, prompt], outputs=[result_ply_file, result_gallery, result_depth])

if __name__ == '__main__':
    demo.launch() 