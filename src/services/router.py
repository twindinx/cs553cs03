# choose which model handles a request and what happens when one fails
# every model call goes through tracked_critique so it's measured

import logging
import os
import gradio as gr
from huggingface_hub import get_token
from src.models.critique import Critique
from src.models.local_model import local_critique
from src.models.remote_model import remote_critique
from src.services.quota import try_use_remote
from src.monitoring.metrics import FAILOVERS, IMAGE_BYTES, REQUESTS
from src.services.tracking import tracked_critique, client_ip

logger = logging.getLogger(__name__)

def _local_with_fallback(image_path, aspect, temperature, top_p, ip) -> Critique:
    # fall back to the remote model if local model fail for unexpected error.
    try:
        return tracked_critique("local", local_critique, image_path, aspect, temperature, top_p)
    except Exception:
        token = get_token()
        # a fallback call also costs token so count it
        if not token or not try_use_remote(ip):
            raise
        # add 1 to photo_critic_failover_total metric to indicate that a failover occurred
        FAILOVERS.labels(reason="local_failed").inc()
        logger.warning("Local model failed, falling back to remote", exc_info=True)
        gr.Warning(
            "The local model is unavailable. "
            "Falling back to the remote model."
        )
        result = tracked_critique("remote", remote_critique, image_path, aspect, temperature, top_p, token)
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
        # add 1 to {model="none", outcome="no_image"} to indicate that a request was made without an image
        REQUESTS.labels(model="none", outcome="no_image").inc()
        raise gr.Error("Please upload a photo before requesting a critique.")

    # record the size of the uploaded image in bytes
    IMAGE_BYTES.observe(os.path.getsize(image_path))
    ip = client_ip(request)

    try:
        # route using local model (and fall back to remote if needed)
        if use_local_model:
            result = _local_with_fallback(image_path,aspect,temperature,top_p, ip)
            return result.to_markdown(), result.to_status()

        # check if token available, if no then use local model
        # use our team token
        token = get_token()
        if not token:
            FAILOVERS.labels(reason="no_token").inc()
            gr.Warning(
                "No Hugging Face credentials available. "
                "Automatically switching to local model."
            )
            result = tracked_critique("local", local_critique, image_path, aspect, temperature, top_p)
            result.route = "Failover (no credentials)"
            return result.to_markdown(), result.to_status()

        # if token ok then use remote model
        # but check the usage first
        if not try_use_remote(ip):
            FAILOVERS.labels(reason="daily_limit").inc()
            gr.Warning(
                "Today's limit for the remote model is reached. "
                "Automatically switching to local model."
            )
            result = tracked_critique("local", local_critique, image_path, aspect, temperature, top_p)
            result.route = "Failover (daily limit reached)"
            return result.to_markdown(), result.to_status()

        # go with remote model when 1) token ok and 2) under daily limit
        try:
            result = tracked_critique("remote", remote_critique, image_path, aspect, temperature, top_p, token)
            return result.to_markdown(), result.to_status()
        except Exception:
            FAILOVERS.labels(reason="remote_failed").inc()
            logger.warning("Remote model failed, failing over to local", exc_info = True)
            gr.Warning(
                "The remote model is unavailable. "
                "Automatically switching to the local model"
            )
            result = tracked_critique("local", local_critique, image_path, aspect, temperature, top_p)
            result.route = "Failover (remote unavailable)"
            return result.to_markdown(), result.to_status()

    except gr.Error:
        raise
    except Exception as error:
        logger.exception("critique failed")
        raise gr.Error(f"The model could not score this photo: {error}") from error



