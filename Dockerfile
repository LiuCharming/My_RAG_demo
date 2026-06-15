FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/vector_db /app/RAG/uploads /app/huggingface_models /app/.hf_cache

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "RAG/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--server.fileWatcherType", "none"]
