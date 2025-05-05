# FILE NAME : ui/tab2.py

import gradio as gr
from ui.utils import list_prompts, list_models, generate_captions_bulk, restart_python

def tab2_ui():
    with gr.Tab("Bulk Folder"):
        folder_input = gr.Textbox(label="Path Folder Gambar")
        prompt_list2 = gr.Dropdown(choices=list_prompts(), label="Pilih Prompt", interactive=True)
        model_list2 = gr.Dropdown(choices=list_models(), label="Pilih Model")

        with gr.Row():
            generate_btn2 = gr.Button("Generate Caption for Folder")
            refresh_btn2 = gr.Button("Refresh Prompt")
            restart_btn2 = gr.Button("Restart Python")

        output_caption2 = gr.Textbox(label="Hasil Proses")

        generate_btn2.click(fn=generate_captions_bulk, inputs=[folder_input, prompt_list2, model_list2], outputs=output_caption2)
        refresh_btn2.click(fn=list_prompts, inputs=None, outputs=prompt_list2)
        restart_btn2.click(fn=restart_python, inputs=None, outputs=None)
