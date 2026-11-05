FROM python:3.12-slim AS builder

# Build-only dependencies: compilers and headers are kept out of the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

# ffmpeg is required by faster-whisper/funasr; sox/sndfile are used by funasr.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsox3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Model caches are mounted from the existing transcribe data volume.
ENV HF_HOME=/data/cache/huggingface
ENV MODELSCOPE_CACHE=/data/cache/modelscope
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

COPY src/ ./src/

# Lower CPU scheduling priority, matching the existing transcribe container.
CMD ["nice", "-n", "15", "python", "src/bot.py"]
