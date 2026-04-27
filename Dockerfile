FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "chatbot.py"]