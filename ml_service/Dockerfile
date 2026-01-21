# syntax = docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For the advanced detector, these are needed.
# We install build-essential to compile mmcv-full, and also add runtime dependencies for OpenCV.
# The build-time dependencies are purged, but the runtime ones are kept.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir anime-face-detector && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
