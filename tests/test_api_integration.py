"""
Integration Tests for API Endpoints.

Tests all major API endpoints and their interactions.
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
try:
    from src.api.server import app
    client = TestClient(app)
except ImportError as e:
    pytest.skip(f"API server not available: {e}", allow_module_level=True)


class TestAlgorithmAPI:
    """Test algorithm management API."""
    
    def test_list_algorithms(self):
        """Test listing algorithms."""
        response = client.get("/algorithms/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "algorithms" in data
        assert "total" in data
        assert data["total"] > 0
    
    def test_get_algorithm_details(self):
        """Test getting algorithm details."""
        response = client.get("/algorithms/0")
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "name" in data
    
    def test_list_categories(self):
        """Test listing algorithm categories."""
        response = client.get("/algorithms/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data


class TestModulationAPI:
    """Test modulation matrix API."""
    
    def test_list_sources(self):
        """Test listing modulation sources."""
        response = client.get("/modulation/sources")
        assert response.status_code == 200
        
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0
    
    def test_list_targets(self):
        """Test listing modulation targets."""
        response = client.get("/modulation/targets")
        assert response.status_code == 200
        
        data = response.json()
        assert "targets" in data
        assert len(data["targets"]) > 0
    
    def test_list_routes(self):
        """Test listing modulation routes."""
        response = client.get("/modulation/routes")
        assert response.status_code == 200
        
        data = response.json()
        assert "routes" in data
        assert "total" in data


class TestConfigAPI:
    """Test configuration management API."""
    
    def test_list_configurations(self):
        """Test listing configurations."""
        response = client.get("/configs/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "configurations" in data
        assert "total" in data


class TestExternalModelsAPI:
    """Test external models API."""
    
    def test_list_gpus(self):
        """Test listing GPUs."""
        response = client.get("/external/gpus/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "devices" in data
        assert "n_devices" in data
    
    def test_list_models(self):
        """Test listing external models."""
        response = client.get("/external/models/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data


class TestMLAPI:
    """Test ML integration API."""
    
    def test_list_ml_models(self):
        """Test listing ML models."""
        response = client.get("/ml/models/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data
    
    def test_get_available_frameworks(self):
        """Test getting available frameworks."""
        response = client.get("/ml/frameworks/available")
        assert response.status_code == 200
        
        data = response.json()
        assert "frameworks" in data
        assert "pytorch" in data["frameworks"]
        assert "tensorflow" in data["frameworks"]
        assert "huggingface" in data["frameworks"]
    
    def test_get_model_types(self):
        """Test getting model types."""
        response = client.get("/ml/model-types")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_types" in data
        assert len(data["model_types"]) > 0


class TestPluginAPI:
    """Test plugin system API."""
    
    def test_list_plugins(self):
        """Test listing plugins."""
        response = client.get("/plugins/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "plugins" in data
        assert "total" in data
    
    def test_list_categories(self):
        """Test listing plugin categories."""
        response = client.get("/plugins/categories/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data
    
    def test_get_statistics(self):
        """Test getting plugin statistics."""
        response = client.get("/plugins/statistics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_plugins" in data
    
    def test_get_examples(self):
        """Test getting plugin examples."""
        response = client.get("/plugins/examples")
        assert response.status_code == 200
        
        data = response.json()
        assert "examples" in data


class TestSessionAPI:
    """Test session management API."""
    
    def test_list_sessions(self):
        """Test listing sessions."""
        response = client.get("/session/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "sessions" in data
        assert "total" in data


class TestTHRMLAPI:
    """Test THRML advanced features API."""
    
    def test_get_temperature(self):
        """Test getting THRML temperature."""
        response = client.get("/thrml/temperature")
        assert response.status_code == 200
        
        data = response.json()
        assert "temperature" in data
    
    def test_list_interactions(self):
        """Test listing higher-order interactions."""
        response = client.get("/thrml/interactions/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "interactions" in data
        assert "total" in data
    
    def test_list_factors(self):
        """Test listing custom factors."""
        response = client.get("/thrml/factors/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "factors" in data
        assert "total" in data
    
    def test_get_factor_library(self):
        """Test getting factor library."""
        response = client.get("/thrml/factors/library")
        assert response.status_code == 200
        
        data = response.json()
        assert "factor_types" in data


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

