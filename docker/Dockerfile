FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOICELAB_ENABLE_TTS=false \
    WEBAPP_HOST=0.0.0.0 \
    WEBAPP_PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY . /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${WEBAPP_PORT}/health || exit 1

WORKDIR /app/webapp
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7860"]

