# FILE NAME : ui/tab2.py

import gradio as gr
from ui.utils import list_prompts, list_models, generate_captions_bulk, restart_python, baca_file
from libs.LoadModel import TextToImage

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

        def load_prompt_contentx(modelx, prompt_filename, image_path):
            prompt_text = baca_file(prompt_filename)  # Membaca konten file prompt
            hasil_caption = TextToImage(modelx, prompt_text, image_path, False)
            return hasil_caption

        generate_btn2.click(fn=load_prompt_contentx, inputs=[model_list2, prompt_list2, folder_input], outputs=output_caption2)
        refresh_btn2.click(fn=list_prompts, inputs=None, outputs=prompt_list2)
        restart_btn2.click(fn=restart_python, inputs=None, outputs=None)
