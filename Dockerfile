FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir runpod==1.7.9 hf_xet && \
    pip install --no-cache-dir -r requirements.txt

# Warm the base model into the volume cache on first boot only; harmless no-op later.
CMD ["python", "-u", "handler.py"]
