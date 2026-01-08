FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt /app/requirements-app.txt

RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.0.1+cu117 \
        torchvision==0.15.2+cu117 \
        --index-url https://download.pytorch.org/whl/cu117 \
    && pip install --no-cache-dir -r /app/requirements-app.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8000

EXPOSE 8000

CMD ["python3", "app.py"]
