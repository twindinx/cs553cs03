import logging
import gradio as gr
from gradio.themes import Soft
from src.models.local_model import load_model
from src.ui.app import demo, CSS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # load (and on a fresh VM, download) the local model before the UI is up
    # NO_RELOAD skips this when `gradio main.py` hot-reloads after a save,
    # the model then loads on the next local critique instead
    if gr.NO_RELOAD:
        load_model()
    demo.launch(css=CSS, theme=Soft())
