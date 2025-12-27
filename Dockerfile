FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY README.md model.py bot.py requirements.txt ./

COPY faiss_index ./faiss_index
COPY readme_screenshots ./readme_screenshots
COPY data ./data


CMD ["python", "bot.py"]
