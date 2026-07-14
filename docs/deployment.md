# Production Deployment & Configuration Guide

This document describes how to deploy the **Real-Time Industrial Defect Detection System** API service to production using Docker, Docker Compose, and how to configure key security and observability metrics.

---

## 1. Production Docker Deployment

The application features a optimized multi-stage `Dockerfile` and a multi-container `docker-compose.yml` orchestrating the FastAPI server, Prometheus metrics, and Grafana visualization.

### Build and Launch Containers
To build and start all production services in detached mode:

```bash
# Build custom images and spin up containers
docker-compose up -d --build

# Verify running container health
docker-compose ps
```

### Port Mappings
The services are exposed on the following default host ports:
* **FastAPI application gateway**: `http://localhost:8000`
* **Prometheus scraper**: `http://localhost:9090`
* **Grafana analytics dashboard**: `http://localhost:3000` (Default credentials: `admin` / `admin`)

---

## 2. Environment Configurations Validation

The system performs automatic safety validations on startup using `validate_production_config()` in `backend/app/core/config.py`.

The following parameters are validated:
1. **API Key Security**: Checks if the `API_KEY` is still using the default fallback key (`industrial-defect-secret-key`). If so, a security warning is logged.
2. **Directory Permissions**: Verifies that the logs directory (`logs/`) and uploads folder (`backend/uploads/`) are fully writable by the `appuser` running the process.
3. **Model Weights presence**: Assures model weights are present at the configured path, falling back to mock predictions if weights are missing.

---

## 3. API Authentication Configuration

To secure predictions in production:
1. Generate a strong API key.
2. Configure it as an environment variable `API_KEY` in the container configuration (via `.env` or Docker Compose environment variables).

```bash
# Example env override
API_KEY=my-extremely-strong-prod-api-key-1234
```

### Consuming Secure Prediction APIs
Requests to prediction endpoints (`/predict/image`, `/predict/video`, `/predict/live`) must include the authentication token in one of two formats:

#### A. Header (Recommended)
Add the header `X-API-Key`:
```http
POST /predict/image HTTP/1.1
Host: localhost:8000
X-API-Key: my-extremely-strong-prod-api-key-1234
Content-Type: multipart/form-data
```

#### B. Bearer Authorization
Add the header `Authorization: Bearer <key>`:
```http
POST /predict/image HTTP/1.1
Host: localhost:8000
Authorization: Bearer my-extremely-strong-prod-api-key-1234
Content-Type: multipart/form-data
```

---

## 4. Production Logging & Telemetry

Production logs are formatted in structured **JSON** format inside `logs/app.log`. This structured format makes it easy to integrate with logging aggregates (e.g. ELK, Loki) and monitor unauthorized attempts.
Log files rotate automatically when they reach **5MB**, keeping up to 3 backups.
