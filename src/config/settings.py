from pathlib import Path

# declare AI model
REMOTE_MODEL = "Qwen/Qwen3.8-27B"
# change to small CPU-friendly VLM because ArtiMuse (15.9 GB) does not fit the VM
LOCAL_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Select model provider
REMOTE_PROVIDER = "auto"

# daily limits on remote model calls, our team's HF Token pays for them
REMOTE_LIMIT_PER_IP = 20 # per visitor (IP address) per day
REMOTE_LIMIT_TOTAL = 200 # for all visitors per day
# keep the count outside repo folder so a redeploy won't reset them
USAGE_DB_PATH = Path.home() / ".photo_critic" / "usage.db"

# define max tokens
REMOTE_MAX_TOKENS = 2048
LOCAL_MAX_TOKENS = 160

# both models have to output same markdown shape
# so UI can us them identically
# heading advice for both models
EVALUATION_HEADING = "How to improve"

# Declare the aspects of the image to evaluate
ASPECTS = [
    "Overall Gestalt",
    "Composition & Design",
    "Visual Elements & Structure",
    "Technical Execution",
    "Originality & Creativity",
]

# Small local model can't output expected result in 1 promp
# so we ask it two simple questions and build the markdown ourselves
LOCAL_SCORE_QUESTION = "Rate the aesthetic quality of this photo from 0 to 100. Answer with only the number."

def local_advice_question(aspect: str) -> str:
    focus = f" in terms of {aspect}" if aspect else ""
    return (f"Suggest three specific ways to improve this photo{focus}. "
            "Answer as exactly three short bullet points, each under 15 words. Do not describe the photo.")

# Ask remote model for score and advice in one reply
def critique_prompt(aspect: str) -> str:
    focus = f" in terms of {aspect}" if aspect else ""
    return f"""Rate this image's aesthetic quality.
Reply in exactly this format, once, and nothing else:

## Score: N / 100

### {EVALUATION_HEADING}
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>

Rules:
- N is a whole number from 0 to 100, judging the image's overall aesthetic quality.
- Do NOT describe what the image shows. The user can already see it.
- Do NOT justify the score or add any prose outside the three bullets.
- Each bullet must name an action the artist can take{focus}.
- Output the format once. Do not repeat it."""