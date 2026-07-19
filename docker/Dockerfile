# Multi-stage build for the tinker-finetune backend.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# The Tinker SDK is optional; uncomment to bake it in for live training:
# RUN pip install tinker

COPY configs ./configs
COPY examples ./examples

EXPOSE 8000

# Default to dry-run so the container is safe to start without credentials.
ENV TF_DRY_RUN=true \
    TF_HOST=0.0.0.0 \
    TF_PORT=8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "tinker_finetune.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
