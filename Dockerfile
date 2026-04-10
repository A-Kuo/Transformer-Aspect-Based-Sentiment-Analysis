# BERT aspect-based sentiment analysis — reproducible CPU/GPU image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: show CLI help (override with e.g. predict / train)
CMD ["python", "main.py", "--help"]
