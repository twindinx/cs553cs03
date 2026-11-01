import re
from dataclasses import dataclass
from src.config.settings import EVALUATION_HEADING

SCORE_PATTERN = re.compile(r"##\s*Score:\s*(\d+)\s*/\s*100")
HEADING_PATTERN = re.compile(rf"###\s*{re.escape(EVALUATION_HEADING)}\s*", re.IGNORECASE)

# define shared result format for both models
@dataclass
class Critique:
    score: int | None
    evaluation: str
    model_name: str
    route: str # "Remote", "Local", "Failover (...)"

    def to_markdown(self) -> str:
        heading = (
            f"## Score: {self.score} / 100"
            if self.score is not None
            else "## Score: unavailable"
        )
        return f"{heading}\n\n### {EVALUATION_HEADING}\n{self.evaluation.strip()}"

    def to_status(self) -> str:
        return f"Critique complete · {self.route} · {self.model_name}"

def parse_response(text: str) -> tuple[int | None, str]:
    # split model markdown into (score, evaluation)
    # both models answer critique_prompt
    # notice the output from Qwen3 sometime duplicate (don't know why!)
    # so we take the first part of the output
    headings = list(SCORE_PATTERN.finditer(text))
    if len(headings) > 1:
        text = text[: headings[1].start()]

    score = int(headings[0].group(1)) if headings else None
    if score is not None and not 0 <= score <= 100:
        score = None

    body = HEADING_PATTERN.split(text, maxsplit=1)
    evaluation = body[1].strip() if len(body) > 1 else text.strip()

    repeat = HEADING_PATTERN.search(evaluation)
    if repeat:
        evaluation = evaluation[: repeat.start()].strip()

    return score, evaluation
