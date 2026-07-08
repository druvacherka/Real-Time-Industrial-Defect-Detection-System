# ==============================================================================
# Dockerfile — Real-Time Industrial Defect Detection System API
# ==============================================================================
FROM python:3.12-slim

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and system metric monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy source code and config templates
COPY backend/app /app/app
COPY backend/README.md /app/README.md
COPY configs /app/configs
COPY .env.example /app/.env.example

# Create model and storage directories
RUN mkdir -p /app/models /app/backend/uploads /app/logs

# Expose FastAPI backend port
EXPOSE 8000

# Start server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
