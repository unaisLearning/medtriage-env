FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY medtriage_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY medtriage_env/ ./medtriage_env/
COPY inference.py ./inference.py

ENV MEDTRIAGE_TASK=task1_single_patient
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "medtriage_env.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
