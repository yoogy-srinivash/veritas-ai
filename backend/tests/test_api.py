import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.main import app
from backend.app.core.database import init_db

# Initialize database before tests
init_db()

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "Veritas AI"
    assert "version" in data
    assert data["health_url"] == "/api/health"

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["database_connected"] is True
    assert data["device"] == "cpu"
    assert data["temperature"] == 1.0
    assert data["rejection_threshold"] == 0.70
    assert "timestamp" in data

def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert isinstance(presets, list)
    assert len(presets) >= 20, f"Expected at least 20 demo presets, found {len(presets)}"
    
    first = presets[0]
    assert "id" in first
    assert "text" in first
    assert "image_file" in first
    assert "image_url" in first
    assert first["image_url"].startswith("/demo_images/")
    assert first["ground_truth"] in ["REAL", "FAKE"]
    assert "category" in first
    assert "source_url" in first
    assert "license" in first
    assert "attribution" in first

def test_demo_image_static_serving():
    # Verify that the static mount successfully serves demo images
    response = client.get("/demo_images/jwst_smacs_0723.jpg")
    assert response.status_code == 200
    assert "image" in response.headers.get("content-type", "")

def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_summary" in data
    assert "models_comparison" in data
    assert "selective_classification" in data
    assert data["source"] in ["INITIAL_SPECIFICATION", "EXPERIMENT_EVALUATION"]

def test_history_empty_initially():
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_explanation_not_found():
    response = client.get("/api/explanations/non-existent-job-id")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()

def test_verify_without_models_returns_503():
    # Day 1 test: without Colab trained weights, returns informative 503
    response = client.post(
        "/api/verify",
        data={"text": "NASA James Webb Space Telescope captures deep field.", "fusion_type": "feature_fusion"}
    )
    assert response.status_code == 503
    data = response.json()
    assert "not yet loaded" in data["detail"].lower() or "expected checkpoint" in data["detail"].lower()
