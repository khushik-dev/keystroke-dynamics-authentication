FROM python:3.11-slim

WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy lean serving requirements first for layer caching
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application source and assets
COPY pipeline.py .
COPY app.py .
COPY model/ model/
COPY static/ static/
COPY data/ data/
RUN mkdir -p monitoring

EXPOSE 8000

# Health check to ensure zero-downtime serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
