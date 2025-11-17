# CADlingo Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from base stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/ ./data/
COPY results/ ./results/
COPY docs/ ./docs/
COPY README.md .
COPY EXECUTION_GUIDE.md .

# Create necessary directories
RUN mkdir -p data/outputs data/processed results/models results/plots results/samples

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose ports
# 8501 for Streamlit
# 8000 for FastAPI
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
