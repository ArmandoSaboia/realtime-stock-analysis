# syntax=docker/dockerfile:1.2
FROM python:3.9-slim AS builder

# Combine ENV statements
ENV SLUGIFY_USES_TEXT_UNIDECODE=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser \
    && mkdir -p /home/appuser/.cache/pip \
    && chown -R appuser:appuser /home/appuser/.cache

WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .

USER appuser

# Combine pip installations
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    pip install --user --upgrade pip==24.0 setuptools wheel && \
    pip install --user --no-cache-dir \
    --timeout=1000 \
    --retries=10 \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

FROM python:3.9-slim

ENV SLUGIFY_USES_TEXT_UNIDECODE=yes \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN useradd -m appuser

COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

ENTRYPOINT ["streamlit", "run", "src/visualization/streamlit_dashboard.py"]
