FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Установка Python 3.12 и системных зависимостей
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Копирование и установка основных зависимостей
COPY pyproject.toml uv.lock /app/
RUN uv venv --python 3.12 /app/.venv && uv sync --frozen

# Активация окружения
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# ВАЖНО: Установите build-инструменты ДО копирования yolov13
RUN uv pip install setuptools wheel

# Копирование yolov13
COPY yolov13/ /app/yolov13/

# Editable install yolov13
RUN cd /app/yolov13 && uv pip install -e .

# Копирование остального кода
RUN mkdir -p /app/data/gestures/some_gesture /app/data/processed /app/checkpoints
COPY modules/ /app/modules/
COPY process_gesture.py config.toml /app/
COPY data/annotations /app/data/annotations/

# Копируем чекпоинты
COPY checkpoints/depth_anything_v2_vitl.pth /app/checkpoints/
COPY checkpoints/ppd.pth /app/checkpoints/
COPY checkpoints/yolov13l.pt /app/checkpoints/

RUN mkdir -p /app/data/gestures /app/data/processed

WORKDIR /app
CMD ["python", "process_gesture.py"]
