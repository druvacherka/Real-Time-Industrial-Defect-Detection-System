# ==============================================================================
# Dockerfile — Real-Time Industrial Defect Detection System API
# Optimized for production deployment: 2026-07-13 (Multi-stage & Security)
# ==============================================================================

# ── Stage 1: Build Dependencies ──
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r /app/requirements.txt

# ── Stage 2: Final Runtime ──
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Install runtime library dependencies for OpenCV and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -u 10001 -m -U appuser && \
    mkdir -p /app/models /app/backend/uploads /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Copy installed Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
# Copy source code and configurations
COPY --chown=appuser:appuser backend/app /app/app
COPY --chown=appuser:appuser backend/README.md /app/README.md
COPY --chown=appuser:appuser configs /app/configs
COPY --chown=appuser:appuser .env.example /app/.env.example

# Expose API port
EXPOSE 8000

# Docker health check to verify service status
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
