# FILE NAME : ui/tab1.py
import torch
import gradio as gr
from ui.utils import list_prompts, list_models, restart_python, baca_file
from libs.LoadModel import TextToImage

def tab1_ui():
    with gr.Tab("Single Image"):
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Input Gambar")
            prompt_list1 = gr.Dropdown(choices=list_prompts(), label="Pilih Prompt", interactive=True)
            model_list1 = gr.Dropdown(choices=list_models(), label="Pilih Model")

        with gr.Row():
            generate_btn1 = gr.Button("Generate Caption")
            refresh_btn1 = gr.Button("Refresh Prompt")
            restart_btn1 = gr.Button("Restart Python")

        output_caption1 = gr.Textbox(label="Caption Output")

        def load_prompt_content(model, prompt_filename, image_path):
            prompt_text = baca_file(prompt_filename)
            hasil_caption = TextToImage(model, prompt_text, image_path, True)
            return hasil_caption

        generate_btn1.click(fn=load_prompt_content, inputs=[model_list1, prompt_list1, image_input], outputs=output_caption1)
        refresh_btn1.click(fn=lambda: gr.update(choices=list_prompts()), inputs=None, outputs=prompt_list1)
        restart_btn1.click(fn=restart_python, inputs=None, outputs=None)
