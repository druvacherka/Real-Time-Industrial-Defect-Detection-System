"""
API Endpoints Unit Tests
=========================
Verifies endpoints using FastAPI TestClient.

Author: prajwaledu802-coder
Date: 2026-07-07
"""

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health endpoint returns successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_endpoint():
    """Verify GET / endpoint returns successfully."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data


def test_predict_image_endpoint():
    """Verify POST /predict/image endpoint returns successfully with sample image input."""
    # Create dummy black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["prediction"]["detection_count"] >= 0


def test_predict_video_validation():
    """Verify video input validation restricts unsupported formats."""
    response = client.post(
        "/predict/video",
        files={"file": ("test.gif", b"dummy bytes", "image/gif")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "unsupported_format"


def test_predict_live_validation():
    """Verify live endpoint connection failure handling with invalid stream URL."""
    response = client.post(
        "/predict/live",
        json={"source": "invalid_source_url", "conf_threshold": 0.25}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "http_error"
