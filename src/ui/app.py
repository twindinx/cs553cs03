import gradio as gr
from src.config.settings import ASPECTS, REMOTE_MODEL, LOCAL_MODEL
from src.utils.formatting import preview_upload
from src.services.router import score_artwork
from gradio.themes import Soft


def clear_workspace():
    return None, None, None, "Upload an image to begin.", "Your critique will appear here."


CSS = """
.gradio-container {
    width: min(1180px, 96%) !important;
    margin: 0 auto !important;
}
#app-title { text-align: center; margin-bottom: 0; }
#app-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin: 4px auto 22px;
}
.workspace-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.07);
}
#critique-output {
    min-height: 430px;
    padding: 20px;
    border-radius: 12px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
}
#status-line { color: var(--body-text-color-subdued); font-size: 0.9rem; }
#model-note { color: var(--body-text-color-subdued); font-size: 0.86rem; }
@media (max-width: 768px) {
    .workspace-panel { padding: 10px; }
    #critique-output { min-height: 260px; }
}
"""

with gr.Blocks(title="Photo Critique") as demo:
    image_path_state = gr.State()

    with gr.Sidebar(width=420):
        gr.Markdown("### Scoring settings")
        use_local_model = gr.Checkbox(
        label="Switch to local model",
        value=False,
        info=f"Runs {LOCAL_MODEL} instead of the remote API."
        )
        aspect = gr.Dropdown(
        label="Aspect to critique",
        choices=ASPECTS,
        value=ASPECTS[0],
        info="Select the aspect of the image you want the model to give advice on."
        )
        temperature = gr.Slider(
        label="Creative freedom (temperature)",
        minimum=0.0,
        maximum=1.5,
        value=0.0,
        step=0.1,
        )
        top_p = gr.Slider(
        label="Diversity (top-p)",
        minimum=0.0,
        maximum=1.0,
        value=1.0,
        step=0.05,
        )
        gr.Markdown("### Model notes")
        gr.Markdown(
        f"Remote model: {REMOTE_MODEL} - a general vision LLM. \n"
        "Remote model daily limit: 20 critiques per day.\n\n"
        f"Local model: {LOCAL_MODEL} - a small VLM model light enough to run in VM.",
        elem_id="model-note"
        )

    gr.Markdown("# 📸 Photo Critique", elem_id="app-title")
    gr.Markdown(
        "Welcome to our photo critique app. "
        "Upload a photo and receive an aesthetic score out of 100, plus concrete suggestions for improving it.", 
        elem_id="app-subtitle")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="workspace-panel"):
            gr.Markdown("### Upload your photo")
            upload_button = gr.UploadButton(
                "Upload an image",
                file_types=["image"],
                file_count="single",
                variant="primary",
            )
            image_preview = gr.Image(label="Photo preview",
            interactive=False, height=360)
            upload_status = gr.Markdown("Upload an image to begin.",
            elem_id="status-line")

            with gr.Row():
                score_button = gr.Button("Score photo", variant="primary", scale=3)
                clear_button = gr.Button("Clear", scale=1)

        with gr.Column(scale=6, elem_classes="workspace-panel"):
            gr.Markdown("### Critique")
            critique_output = gr.Markdown(
                "Your critique will appear here.",
                elem_id="critique-output",
            )
            model_status = gr.Markdown("", elem_id="status-line")

    upload_button.upload(
        fn=preview_upload,
        inputs=upload_button,
        outputs=[image_preview, image_path_state, upload_status],
    )

    score_button.click(
        fn=score_artwork,
        inputs=[
            image_path_state,
            aspect,
            temperature,
            top_p,
            use_local_model,
        ],
        outputs=[critique_output,model_status],
    )

    clear_button.click(
        fn=clear_workspace,
        outputs=[
            upload_button,
            image_preview,
            image_path_state,
            upload_status,
            critique_output,
        ],
    ).then(lambda: "", outputs=model_status)
    demo.load()

if __name__ == "__main__":
    demo.launch(css=CSS, theme=Soft())