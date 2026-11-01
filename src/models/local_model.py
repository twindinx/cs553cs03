import logging
import re
from typing import Any
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config.settings import LOCAL_MODEL, LOCAL_MAX_TOKENS, LOCAL_SCORE_QUESTION, local_advice_question
from src.models.critique import Critique

logger = logging.getLogger(__name__)

# run a small model on CPU in fp32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

NUMBER_PATTERN = re.compile(r"\d+")
# split "1. a 2. b", "- a\n- b" or "* a" into separate advices
ADVICE_SPLIT_PATTERN = re.compile(r"(?:^|\s)(?:\d+[.)]|[-*•])\s+")

_model = None
_processor = None


def load_model():
    # load model once at startup and cache
    global _model, _processor
    if _model is not None:
        return _model, _processor

    logger.info("Loading %s on %s (%s)", LOCAL_MODEL, DEVICE, DTYPE)
    _processor = AutoProcessor.from_pretrained(LOCAL_MODEL, do_image_splitting=False)
    _model = AutoModelForImageTextToText.from_pretrained(LOCAL_MODEL, dtype=DTYPE, device_map=DEVICE)
    return _model, _processor


def parse_score(text: str) -> int | None:
    # take the first number the model says, e.g. '80.' -> 80
    match = NUMBER_PATTERN.search(text)
    if not match:
        return None
    score = int(match.group())
    return score if 0 <= score <= 100 else None


def format_advices(text: str) -> str:
    # format the model's list into markdown bullets, keep at most three
    parts = ADVICE_SPLIT_PATTERN.split(text)
    # drop any preamble before the first bullet, e.g. "Here are three ways:"
    if len(parts) > 1:
        parts = parts[1:]
    advices = [advice.strip() for advice in parts if advice.strip()]
    return "\n".join(f"- {advice}" for advice in advices[:3])


def _ask(model, processor, image, question, gen_config) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE, DTYPE)

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_config)

    # drop the prompt tokens and keep only the generated answer
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def local_critique(image_path, aspect, temperature, top_p) -> Critique:
    model, processor = load_model()

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    # get the score
    score_config = dict(max_new_tokens=8, do_sample=False)
    raw_score = _ask(model, processor, image, LOCAL_SCORE_QUESTION, score_config)
    score = parse_score(raw_score)
    if score is None:
        logger.warning("Local model returned an unparseable score: %r", raw_score)

    # get the advices
    advice_config: dict[str, Any] = dict(max_new_tokens=LOCAL_MAX_TOKENS, do_sample=temperature > 0)
    if temperature > 0:
        advice_config.update(temperature=float(temperature), top_p=float(top_p))
    advice = _ask(model, processor, image, local_advice_question(aspect), advice_config)
    if not advice:
        raise RuntimeError("Local model returned an empty response.")

    return Critique(
        score=score,
        evaluation=format_advices(advice),
        model_name=LOCAL_MODEL,
        route="Local",
    )
