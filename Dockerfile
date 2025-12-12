FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        libomp-dev \
        wget \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Проброс модели sentence-transformers в кеш
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
# ENV HF_TOKEN=...

CMD ["python", "bot.py"]
