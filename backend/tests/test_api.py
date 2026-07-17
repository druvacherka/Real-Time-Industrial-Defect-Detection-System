"""
API Endpoints Unit Tests
=========================
Verifies endpoints using FastAPI TestClient and tests API Key authentication middleware.

Author: prajwaledu802-coder
Date: 2026-07-14
"""

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

@pytest.fixture(name="client")
def client_fixture():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verify GET /health endpoint returns successfully without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "disk_percent" in data
    assert isinstance(data["disk_percent"], float)
    assert 0.0 <= data["disk_percent"] <= 100.0


def test_root_endpoint(client):
    """Verify GET / endpoint returns successfully without auth."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data


def test_predict_image_unauthorized(client):
    """Verify prediction endpoints return 401 when X-API-Key is missing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_predict_image_forbidden(client):
    """Verify prediction endpoints return 403 when X-API-Key is incorrect."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", img_bytes, "image/png")},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_predict_image_endpoint(client):
    """Verify POST /predict/image endpoint returns successfully with correct API Key."""
    # Create dummy black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", img_bytes, "image/png")},
        headers={"X-API-Key": settings.API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["prediction"]["detection_count"] >= 0


def test_predict_video_validation(client):
    """Verify video input validation restricts unsupported formats (with auth)."""
    response = client.post(
        "/predict/video",
        files={"file": ("test.gif", b"dummy bytes", "image/gif")},
        headers={"X-API-Key": settings.API_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "unsupported_format"


def test_predict_live_validation(client):
    """Verify live endpoint connection failure handling with invalid stream URL (with auth)."""
    response = client.post(
        "/predict/live",
        json={"source": "invalid_source_url", "conf_threshold": 0.25},
        headers={"X-API-Key": settings.API_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "http_error"
