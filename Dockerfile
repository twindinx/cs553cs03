FROM python:3.14-slim

# WORKDIR /opt/app
# COPY . .
# RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# install packages that we need
# noninteractive mode to avoid pausing for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends prometheus-node-exporter curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY main.py ./
COPY src ./src
COPY docker/entrypoint.sh /entrypoint.sh

# setting the environment variable
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    METRICS_PORT=8000 \
    HF_HOME=/data/hf

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100

HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -fsS http://localhost:7860 || exit 1

ENTRYPOINT ["/entrypoint.sh"]
