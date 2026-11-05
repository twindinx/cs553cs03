import logging
import os
import time

import gradio as gr
from src.monitoring.metrics import IN_PROGRESS, LATENCY, REQUESTS, SCORES


def client_ip(request: gr.Request) -> str:
    # get the client IP address from the request headers
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # if no X-Forwarded-For header, get the IP address from the request context
    return request.client.host if request.client else "unknown"

# track the critique
def tracked_critique(model, critique_fn, image_path, *args):
    # raise the in-progress gauge from 0 to 1 to indicate that a critique is in progress
    IN_PROGRESS.inc()

    start = time.perf_counter()
    try:
        result = critique_fn(image_path, *args)
    except Exception:
        REQUESTS.labels(model=model, outcome="error").inc()
        raise
    finally:
        # run whether the call worked or failed
        IN_PROGRESS.dec()

    # measure the latency of the critique function
    LATENCY.labels(model=model).observe(time.perf_counter() - start)
    REQUESTS.labels(model=model, outcome="success").inc()

    if result.score is not None:
        SCORES.labels(model=model).observe(result.score)
    # return the result to the UI
    return result



    