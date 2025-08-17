# Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY README.md ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Note: LM Studio runs on the HOST at port 1234.
ENV LMSTUDIO_BASE_URL="http://host.docker.internal:1234/v1"

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]