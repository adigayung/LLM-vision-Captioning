# FILE NAME : main.py

import gradio as gr
import warnings

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

app.launch(share=IN_COLAB)
