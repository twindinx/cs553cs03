import time
from prometheus_client import Counter, Histogram
import base64
import io
from PIL import Image

from huggingface_hub import InferenceClient

from src.config.settings import REMOTE_MODEL, REMOTE_PROVIDER, REMOTE_MAX_TOKENS, critique_prompt
from src.models.critique import Critique, parse_response

REMOTE_SECONDS = Histogram(
    "photo_critic_remote_call_seconds", "Time waiting for the remote model",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)
REMOTE_ERRORS = Counter(
    "photo_critic_remote_errors_total", "Failures calling the remote model",["error_type"]
)
REMOTE_TOKENS = Counter(
    "photo_critic_remote_tokens_total", "Tokens used by the remote model", ["kind"])
MAX_IMAGE_SIDE = 768

def image_as_data_url(image_path: str) -> str:
    # downscale image size before sending to model to reduce tokens
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

def remote_critique(image_path, aspect, temperature, top_p, hf_token) -> Critique:
    client = InferenceClient(provider=REMOTE_PROVIDER, token=hf_token)
    image_url = image_as_data_url(image_path)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model = REMOTE_MODEL,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": critique_prompt(aspect)},
                    ],
                }
            ],
            max_tokens= REMOTE_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as error:
        REMOTE_ERRORS.labels(error_type=type(error).__name__).inc()
        raise
    finally:
        REMOTE_SECONDS.observe(time.perf_counter() - start)
    
    usage = getattr(response, "usage", None)
    if usage:
        REMOTE_TOKENS.labels(kind="prompt").inc(usage.prompt_tokens or 0)
        REMOTE_TOKENS.labels(kind="completion").inc(usage.completion_tokens or 0)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    # catch the case where reasoning exhaust the token budget
    if not content:
        REMOTE_ERRORS.labels(error_type="empty_response").inc()
        raise RuntimeError(
            f"Remote model returned no visible content (finish reasons="
            f"{choice.finish_reason}); the token budget was likely exhausted by internal reasoning."
        )

    score, evaluation = parse_response(content)
    return Critique(score=score, evaluation=evaluation, model_name=REMOTE_MODEL, route="Remote")
