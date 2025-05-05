# FILE NAME : ui/tab3.py

import gradio as gr
from ui.utils import (
    list_prompts, get_file_content, save_file_content,
    create_new_file, delete_file, restart_python
)

def tab3_ui():
    with gr.Tab("Edit Prompt"):
        file_list = gr.Dropdown(choices=list_prompts(), label="Pilih File Prompt", interactive=True)
        refresh_btn3 = gr.Button("Refresh Prompt List")
        restart_btn3 = gr.Button("Restart Python")

        text_editor = gr.Textbox(label="Isi File", lines=20)
        save_btn = gr.Button("Simpan Perubahan")
        notif = gr.Textbox(label="Status")

        file_list.change(fn=get_file_content, inputs=file_list, outputs=text_editor)
        save_btn.click(fn=save_file_content, inputs=[file_list, text_editor], outputs=notif)
        refresh_btn3.click(fn=list_prompts, inputs=None, outputs=file_list)
        restart_btn3.click(fn=restart_python, inputs=None, outputs=None)

        with gr.Row():
            new_file_input = gr.Textbox(label="Nama File Baru (akhiri .txt)")
            create_btn = gr.Button("Buat File Baru")
            delete_btn = gr.Button("Hapus File")
        notif_new = gr.Textbox(label="Status File Baru / Hapus")

        create_btn.click(fn=create_new_file, inputs=new_file_input, outputs=notif_new)
        delete_btn.click(fn=delete_file, inputs=file_list, outputs=notif_new)
