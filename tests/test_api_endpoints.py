"""
Test Suite for ML and Training API Endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    from src.api.ml_endpoints import router as ml_router
    from src.api.training_endpoints import router as training_router
    from fastapi import FastAPI
    
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)

# Create test app
app = FastAPI()
app.include_router(ml_router)
app.include_router(training_router)

client = TestClient(app)


# ============================================================================
# ML Endpoints Tests
# ============================================================================

def test_health_check():
    """Test ML health endpoint."""
    response = client.get("/api/ml/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    print("✓ ML health check works")


def test_list_models():
    """Test listing models."""
    response = client.get("/api/ml/models")
    
    # May return 503 if ML not available, or 200 if available
    if response.status_code == 200:
        data = response.json()
        assert "models" in data
        assert "count" in data
        print(f"✓ List models works ({data['count']} models)")
    elif response.status_code == 503:
        print("⊘ ML modules not available (expected in minimal environment)")
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_search_models():
    """Test searching models."""
    response = client.get("/api/ml/models/search?query=bert")
    
    if response.status_code == 200:
        data = response.json()
        assert "models" in data
        assert "query" in data
        print("✓ Search models works")
    elif response.status_code == 503:
        print("⊘ ML modules not available")


def test_get_loaded_models():
    """Test listing loaded models."""
    response = client.get("/api/ml/models/loaded")
    assert response.status_code == 200
    data = response.json()
    assert "loaded_models" in data
    print("✓ List loaded models works")


def test_model_status_not_found():
    """Test getting status of non-existent model."""
    response = client.get("/api/ml/models/nonexistent/status")
    
    # Should return 404 or 503 (if ML not available)
    assert response.status_code in [404, 503]
    print("✓ Model status 404 handling works")


# ============================================================================
# Training Endpoints Tests
# ============================================================================

def test_start_training():
    """Test starting a training session."""
    payload = {
        "session_id": "test_session_1",
        "model_id": "test_model",
        "config": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    response = client.post("/api/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert data["session_id"] == "test_session_1"
    print("✓ Start training works")


def test_get_training_status():
    """Test getting training status."""
    # First start a session
    payload = {
        "session_id": "test_session_2",
        "model_id": "test_model",
        "config": {"epochs": 10}
    }
    client.post("/api/training/start", json=payload)
    
    # Then get status
    response = client.get("/api/training/status/test_session_2")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "status" in data
    print("✓ Get training status works")


def test_stop_training():
    """Test stopping a training session."""
    # Start session
    payload = {
        "session_id": "test_session_3",
        "model_id": "test_model",
        "config": {"epochs": 10}
    }
    client.post("/api/training/start", json=payload)
    
    # Stop session
    response = client.post("/api/training/stop?session_id=test_session_3")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stopped"
    print("✓ Stop training works")


def test_list_training_sessions():
    """Test listing training sessions."""
    response = client.get("/api/training/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert "count" in data
    print(f"✓ List sessions works ({data['count']} sessions)")


def test_get_training_metrics():
    """Test getting training metrics."""
    # Start session
    payload = {
        "session_id": "test_session_4",
        "model_id": "test_model",
        "config": {"epochs": 5}
    }
    client.post("/api/training/start", json=payload)
    
    # Get metrics
    response = client.get("/api/training/metrics/test_session_4")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "history" in data
    print("✓ Get training metrics works")


def test_delete_training_session():
    """Test deleting a training session."""
    # Start session
    payload = {
        "session_id": "test_session_5",
        "model_id": "test_model",
        "config": {"epochs": 5}
    }
    client.post("/api/training/start", json=payload)
    
    # Delete session
    response = client.delete("/api/training/sessions/test_session_5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    print("✓ Delete session works")


def test_training_status_not_found():
    """Test getting status of non-existent session."""
    response = client.get("/api/training/status/nonexistent")
    assert response.status_code == 404
    print("✓ Training status 404 handling works")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("API Endpoints Test Suite")
    print("="*70 + "\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Search Models", test_search_models),
        ("Get Loaded Models", test_get_loaded_models),
        ("Model Status 404", test_model_status_not_found),
        ("Start Training", test_start_training),
        ("Get Training Status", test_get_training_status),
        ("Stop Training", test_stop_training),
        ("List Training Sessions", test_list_training_sessions),
        ("Get Training Metrics", test_get_training_metrics),
        ("Delete Training Session", test_delete_training_session),
        ("Training Status 404", test_training_status_not_found)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)

