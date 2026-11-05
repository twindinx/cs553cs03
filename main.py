import logging
import gradio as gr
from gradio.themes import Soft
from src.models.local_model import load_model
from src.ui.app import demo, CSS
from src.config.settings import LOCAL_MODEL, REMOTE_MODEL
from src.monitoring.metrics import BUILD, start_metrics_server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # load (and on a fresh VM, download) the local model before the UI is up
    # NO_RELOAD skips this when `gradio main.py` hot-reloads after a save,
    # the model then loads on the next local critique instead
    if gr.NO_RELOAD:
        # start metric server before loading the model
        start_metrics_server()
        # fill in BUILD info with model names
        BUILD.info({"remote_model": REMOTE_MODEL, "local_model": LOCAL_MODEL})
        # load model
        load_model()
    demo.launch(css=CSS, theme=Soft())
