FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="Kokoro TTS Matrix Team" \
      description="Kokoro TTS Matrix Service with CUDA support" \
      version="1.0"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        espeak-ng \
        portaudio19-dev \
        libsndfile1 \
        ffmpeg \
        curl \
        build-essential \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN mkdir -p outputs && chmod 777 outputs

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install kokoro>=1.2.0 soundfile && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["python3", "-m", "src.kokoro_matrix.gradio_interface", "--server-name", "0.0.0.0", "--share-name", "localhost", "--reload"]
