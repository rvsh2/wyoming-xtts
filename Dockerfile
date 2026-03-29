FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2 \
    XTTS_DEVICE=cuda \
    XTTS_DEFAULT_LANGUAGE=pl \
    XTTS_DEFAULT_VOICE=default \
    XTTS_SPEAKER_DIR=/data/speakers \
    XTTS_MODEL_DIR=/data/models \
    WYOMING_URI=tcp://0.0.0.0:10201 \
    COQUI_TOS_AGREED=1 \
    HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8180

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install .

RUN mkdir -p /data/models /data/speakers

EXPOSE 10201 8180

CMD ["python", "-m", "xtts_wyoming", "--uri", "tcp://0.0.0.0:10201", "--speaker-dir", "/data/speakers", "--model-dir", "/data/models", "--http-host", "0.0.0.0", "--http-port", "8180"]
