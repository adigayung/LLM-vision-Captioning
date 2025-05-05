
import gradio as gr
import warnings
import webbrowser

warnings.filterwarnings("ignore")

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

from ui.tab1 import tab1_ui
from ui.tab2 import tab2_ui
from ui.tab3 import tab3_ui

with gr.Blocks() as app:
    with gr.Tabs():
        tab1_ui()
        tab2_ui()
        tab3_ui()

if not IN_COLAB:
    webbrowser.open("http://127.0.0.1:7534/?__theme=dark")

# app.launch(share=IN_COLAB)
app.launch(share=IN_COLAB, server_port=7534)


