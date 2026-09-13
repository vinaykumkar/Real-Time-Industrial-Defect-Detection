# Real-Time Industrial Defect Detection System
# Build:  docker build -t defect-detection .
# Run:    docker run -p 8000:8000 defect-detection
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# OpenCV runtime libs for slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch first (container has no GPU): avoids ~3 GB of NVIDIA wheels
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY monitoring ./monitoring
COPY data/yolo/data.yaml ./data/yolo/data.yaml

# model weights, history db and outputs are expected as volumes:
#   -v ./models:/workspace/models -v ./outputs:/workspace/outputs -v ./data/detections.db:/workspace/data/detections.db
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
