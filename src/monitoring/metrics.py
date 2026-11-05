import os

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

def start_metrics_server():
    start_http_server(int(os.getenv("METRICS_PORT", 8000)))

REQUESTS = Counter("photo_critic_requests_total", "Critique requests by model and outcome", ["model", "outcome"])

LATENCY = Histogram("photo_critic_request_duration_seconds", "Time to produce one critique", ["model"],
                    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300))

IN_PROGRESS = Gauge("photo_critic_requests_in_progress", "Number of critique requests in progress")

# get the score that the model gives
SCORES = Histogram("photo_critic_scores", "Scores given by the model", ["model"], 
                   buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

# upload size in bytes, for example 1e5 = 100 KB, upto 2e7 - 20 MB
IMAGE_BYTES = Histogram("photo_critic_image_bytes", "Size of uploaded photos in bytes",
                        buckets = (1e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7))

FAILOVERS = Counter("photo_critic_failovers_total", "Switching to the other model, by reason", ["reason"])

BUILD = Info("photo_critic_build", "Models this app runs")

for model in ("remote", "local"):
    for outcome in ("success", "error"):
        REQUESTS.labels(model=model, outcome=outcome)
    LATENCY.labels(model=model)
    SCORES.labels(model=model)
REQUESTS.labels(model="none", outcome="no_image")

for reason in ("daily_limit", "no_token", "remote_failed", "local_failed"):
    FAILOVERS.labels(reason=reason)
