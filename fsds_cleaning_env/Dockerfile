FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY . /app/env

RUN pip install --no-cache-dir "openenv-core[core]>=0.2.1" && \
    pip install --no-cache-dir /app/env

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/env:$PYTHONPATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
