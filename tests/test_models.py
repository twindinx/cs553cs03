import pytest
from src.models.critique import Critique, parse_response

def test_parse_response_extracts_score_and_body():
    text = "## Score: 58 / 100\n\n### How to improve\n- Crop the foreground."
    score, evaluation = parse_response(text)
    assert score == 58
    assert evaluation == "- Crop the foreground."


def test_parse_response_discards_duplicated_answer():
    # Qwen3 sometimes emits the whole answer twice; keep only the first
    block = "## Score: 58 / 100\n\n### How to improve\n- Crop the foreground.\n"
    score, evaluation = parse_response(block + "\n" + block)
    assert score == 58
    assert evaluation == "- Crop the foreground."


def test_parse_response_without_score_keeps_text():
    score, evaluation = parse_response("The model rambled without a score.")
    assert score is None
    assert evaluation == "The model rambled without a score."