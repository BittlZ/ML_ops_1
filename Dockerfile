FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INPUT_DIR=/app/input \
    OUTPUT_DIR=/app/output \
    MODELS_DIR=/app/models \
    TRAIN_CSV=/app/train.csv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY train.csv ./train.csv

RUN mkdir -p /app/input /app/output
VOLUME ["/app/input", "/app/output"]

CMD ["python", "-m", "src.main"]
