# choose which model handles a request and what happens when one fails

import logging
import gradio as gr
from huggingface_hub import get_token
from src.models.critique import Critique
from src.models.local_model import local_critique
from src.models.remote_model import remote_critique
from src.services.quota import try_use_remote

logger = logging.getLogger(__name__)

def _client_ip(request: gr.Request) -> str:
    # get visitor IP address to count their remote calls
    return request.client.host if request.client else "unknown"

def _local_with_fallback(image_path, aspect, temperature, top_p, ip) -> Critique:
    # fall back to the remote model if local model fail for unexpected error.
    try:
        return local_critique(image_path, aspect, temperature, top_p)
    except Exception:
        token = get_token()
        # a fallback call also costs token so count it
        if not token or not try_use_remote(ip):
            raise
        logger.warning("Local model failed, falling back to remote", exc_info=True)
        gr.Warning(
            "The local model is unavailable. "
            "Falling back to the remote model."
        )
        result = remote_critique(image_path,aspect,temperature,top_p,token)
        result.route = "Failover (local unavailable)"
        return result


def score_artwork(
    image_path,
    aspect,
    temperature,
    top_p,
    use_local_model,
    request: gr.Request,
):
    #  return (markdown, status) for the UI
    if not image_path:
        raise gr.Error("Please upload a photo before requesting a critique.")

    ip = _client_ip(request)
    try:
        # route using local model (and fall back to remote if needed)
        if use_local_model:
            result = _local_with_fallback(image_path,aspect,temperature,top_p, ip)
            return result.to_markdown(), result.to_status()

        # check if token available, if no then use local model
        # use our team token
        token = get_token()
        if not token:
            gr.Warning(
                "No Hugging Face credentials available. "
                "Automatically switching to local model."
            )
            result = local_critique(image_path,aspect,temperature,top_p)
            result.route = "Failover (no credentials)"
            return result.to_markdown(), result.to_status()

        # if token ok then use remote model
        # but check the usage first
        if not try_use_remote(ip):
            gr.Warning(
                "Today's limit for the remote model is reached. "
                "Automatically switching to local model."
            )
            result = local_critique(image_path,aspect,temperature,top_p)
            result.route = "Failover (daily limit reached)"
            return result.to_markdown(), result.to_status()

        # go with remote model when 1) token ok and 2) under daily limit
        try:
            result = remote_critique(image_path, aspect, temperature, top_p, token)
            return result.to_markdown(), result.to_status()
        except Exception:
            logger.warning("Remote model failed, failing over to local", exc_info = True)
            gr.Warning(
                "The remote model is unavailable. "
                "Automatically switching to the local model"
            )
            result = local_critique(image_path, aspect, temperature, top_p)
            result.route = "Failover (remote unavailable)"
            return result.to_markdown(), result.to_status()

    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"The model could not score this photo: {error}") from error



