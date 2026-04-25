# SurviveCity v2 server container — for OpenEnv submission and local serving.
# Training is NOT done in this container; use a torch-cuda image (e.g. nvcr.io
# pytorch:24.01-py3) for DGX training. This image is for running the env-only.

FROM python:3.11-slim

WORKDIR /app

# System deps (slim base lacks gcc for some pydantic/fastapi extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY survivecity_v2_env /app/survivecity_v2_env
COPY server /app/server

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 7861

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7861"]
