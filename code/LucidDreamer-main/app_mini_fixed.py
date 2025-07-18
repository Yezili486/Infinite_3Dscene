# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import glob
import platform
import pathlib
import shlex
import subprocess
import gradio as gr
from huggingface_hub import snapshot_download


root = pathlib.Path(__file__).parent
example_root = os.path.join(root, 'examples')
ckpt_root = os.path.join(root, 'stablediffusion')

user_os = platform.system()
if user_os.lower() == 'windows':
    use_symlinks = False
else:
    use_symlinks = 'auto'

print("Downloading demo examples...")
d = example_root
if len(glob.glob(os.path.join(d, '*.ply'))) < 8:
    try:
        snapshot_download(repo_id="ironjr/LucidDreamerDemo", repo_type="model", local_dir=d, local_dir_use_symlinks=use_symlinks)
        print("Demo examples downloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not download demo examples: {e}")

print("Downloading SD model...")
d = os.path.join(ckpt_root, 'SD1-5')
if not os.path.exists(d):
    try:
        snapshot_download(repo_id="runwayml/stable-diffusion-inpainting", repo_type="model", local_dir=d, local_dir_use_symlinks=use_symlinks)
        print("SD model downloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not download SD model: {e}")

# Skip the compilation steps for now - user can install manually if needed
print("Skipping compilation of simple-knn and depth-diff-gaussian-rasterization-min...")
print("If you need full functionality, please compile these modules manually following the Windows installation guide.")

try:
    from luciddreamer import LucidDreamer
    print("LucidDreamer imported successfully!")
except ImportError as e:
    print(f"Warning: Could not import LucidDreamer: {e}")
    print("Running in demo mode with limited functionality...")
    
    # Create a dummy LucidDreamer class for demo purposes
    class LucidDreamer:
        def __init__(self, save_dir='./'):
            self.save_dir = save_dir
            print("Demo LucidDreamer initialized")
        
        def run(self, *args):
            print("Demo run function called - full functionality requires compiled modules")
            return None, None, None
        
        def create(self, *args):
            print("Demo create function called - full functionality requires compiled modules")
            return None
        
        def render_video(self, *args):
            print("Demo render_video function called - full functionality requires compiled modules")
            return None, None

css = """
#run-button {
  background: coral;
  color: white;
}
"""

ld = LucidDreamer(save_dir='./')

with gr.Blocks(css=css) as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div>
            <h1 >LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes</h1>
            <h5 style="margin: 0;">If you like our project, please visit our Github, too! ✨✨✨ More features are waiting!</h5>
            </br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href='https://arxiv.org/abs/2311.13384'>
                    <img src="https://img.shields.io/badge/Arxiv-2311.13384-red">
                </a>
                &nbsp;
                <a href='https://luciddreamer-cvlab.github.io'>
                    <img src='https://img.shields.io/badge/Project-LucidDreamer-green' alt='Project Page'>
                </a>
                &nbsp;
                <a href='https://github.com/luciddreamer-cvlab/LucidDreamer'>
                    <img src='https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer?label=Github&color=blue'>
                </a>
                &nbsp;
                <a href='https://twitter.com/_ironjr_'>
                    <img src='https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_'>
                </a>
            </div>
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
            model_name = gr.Radio(
                label='SD checkpoint',
                choices=['SD1.5 (default)'],
                value='SD1.5 (default)'
            )
            
            prompt = gr.Textbox(
                label='Text prompt',
                value='A cozy livingroom',
            )
            n_prompt = gr.Textbox(
                label='Negative prompt',
                value='photo frame, frame, boarder, simple color, inconsistent, humans, people',
            )
            gen_camerapath = gr.Radio(
                label='Camera trajectory for generation (STEP 1)',
                choices=['lookaround', 'lookdown', 'rotate360'],
                value='lookaround',
            )
            
            with gr.Row():
                seed = gr.Slider(
                    label='Seed',
                    minimum=1,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
                diff_steps = gr.Slider(
                    label='SD inpainting steps',
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=30,
                )

            render_camerapath = gr.Radio(
                label='Camera trajectory for rendering (STEP 2)',
                choices=['back_and_forth', 'llff', 'headbanging'],
                value='llff',
            )

        with gr.Column():
            run_button = gr.Button(value='Run! (it may take a while)', elem_id='run-button')

            gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                    <h3>...or you can run in two steps</h3>
                    <h5>(hint: press STEP 2 if you have already baked Gaussians in STEP 1).</h5>
                </div>
                </div>
                """
            )

            with gr.Row():
                gaussian_button = gr.Button(value='STEP 1: Generate Gaussians')
                render_button = gr.Button(value='STEP 2: Render A Video')

            gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                    <h5>...or you can just watch a quick preload we have baked already.</h5>
                </div>
                </div>
                """
            )

            example_name = gr.Radio(
                label='Quick load',
                choices=['animelake', 'fantasy', 'kitchen', 'DON\'T'],
                value='DON\'T',
            )

    ips = [example_name, input_image, prompt, n_prompt, gen_camerapath, seed, diff_steps, render_camerapath, model_name]

    run_button.click(fn=ld.run, inputs=ips[1:] + ips[:1], outputs=[result_ply_file, result_gallery, result_depth])
    gaussian_button.click(fn=ld.create, inputs=ips[1:-2] + ips[-1:] + ips[:1], outputs=[result_ply_file])
    render_button.click(fn=ld.render_video, inputs=ips[-2:-1] + ips[:1], outputs=[result_gallery, result_depth])

    gr.Examples(
        examples=[
            [
                'animelake',
                'examples/Image015_animelakehouse.jpg',
                'anime style, animation, best quality, a boat on lake, trees and rocks near the lake. a house and port in front of a house',
                'photo frame, frame, boarder, simple color, inconsistent',
                'lookdown',
                1,
                50,
                'back_and_forth',
                'SD1.5 (default)',
            ],
        ],
        inputs=ips,
        outputs=[result_ply_file, result_gallery, result_depth],
        fn=ld.run,
        cache_examples=False,
    )

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: left;">
        </br>
        <div>
            <h5 style="margin: 0;">Status: Running with limited functionality</h5>
            </br>
            <p>Some features may be unavailable due to missing compiled modules. For full functionality, please follow the Windows installation guide to compile simple-knn and depth-diff-gaussian-rasterization-min.</p>
        </div>
        </div>
        """
    )


if __name__ == '__main__':
    demo.launch() 