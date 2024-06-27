import numpy as np
import gradio as gr
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image


def swap_face(source_file, target_file, doFaceEnhancer):

    source_path = "input.jpg"
    target_path = "target.jpg"

    source_image = Image.fromarray(source_file)
    source_image.save(source_path)
    target_image = Image.fromarray(target_file)
    target_image.save(target_path)

    print("source_path: ", source_path)
    print("target_path: ", target_path)

    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    output_path = "output.jpg"
    roop.globals.output_path = normalize_output_path(
        roop.globals.source_path, roop.globals.target_path, output_path
    )
    if doFaceEnhancer:
        roop.globals.frame_processors = ["face_swapper", "face_enhancer"]
    else:
        roop.globals.frame_processors = ["face_swapper"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()

    print(
        "start process",
        roop.globals.source_path,
        roop.globals.target_path,
        roop.globals.output_path,
    )

    for frame_processor in get_frame_processors_modules(
        roop.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return

    start()
    return output_path


html_section_1 = "<div><h1>Welcome to the Face Swap App</h1></div>"
html_section_2 = "<div><p>Upload your source and target images to swap faces. Optionally, use the face enhancer feature.</p></div>"
html_section_3 = """<div>
    <a href="https://ziverr.xyz/monica" target="_blank" style="display: inline-block;">
        <img decoding="async" alt="banner" src="https://ziverr.xyz/wp-content/uploads/2024/06/PASSIVE-3.gif">
    </a>
    <a href="https://go.fiverr.com/visit/?bta=36184&brand=fiverrcpa&landingPage=https%253A%252F%252Fwww.fiverr.com%252Fcategories%252Fprogramming-tech%252Fai-coding%252Fai-applications%253Fsource%253Dcategory_tree" target="_blank" style="display: inline-block;">
        <img fetchpriority="high" decoding="async" width="468" height="120" src="https://ziverr.xyz/wp-content/uploads/2024/06/PASSIVE-1.gif" class="attachment-large size-large wp-image-1266" alt="">
    </a>
    <a href="https://beta.publishers.adsterra.com/referral/UNXJYTziBP" target="_blank" style="display: inline-block;">
        <img decoding="async" alt="banner" src="https://landings-cdn.adsterratech.com/referralBanners/gif/468x120_adsterra_reff.gif">
    </a>
</div>"""

app = gr.Blocks()

with app:
    gr.HTML(html_section_1)
    gr.HTML(html_section_2)
    gr.HTML(html_section_3)
    gr.Interface(
        fn=swap_face,
        inputs=[gr.Image(), gr.Image(), gr.Checkbox(label="face_enhancer?", info="do face enhancer?")],
        outputs="image"
    )

app.launch()
