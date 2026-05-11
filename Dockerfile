FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

# Copy all project files (includes logo.png, index1.html, embeddings.pkl, etc.)
COPY . .

# Expose the port
EXPOSE 8000

# Health check so Docker knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with hot reload for dev; remove --reload for production
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]