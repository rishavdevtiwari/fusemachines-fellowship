# ==============================================================================
# Multi-Stage Production Dockerfile for ShopAssist AI (Task 1 Deliverable)
# Packages FastAPI Backend, Streamlit Web UI, RAG Pipeline, and ONNX Runtime
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# --- Final Production Image ---
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code and models
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY requirements.txt /app/
COPY README.md /app/

# Create non-root system user for container security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API (8000) and Web UI (8501) ports
EXPOSE 8000 8501

# Health check against FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default Command: Start FastAPI Backend
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
