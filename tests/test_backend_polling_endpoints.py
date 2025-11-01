import types
import sys
from fastapi.testclient import TestClient
import importlib

# Provide a lightweight prometheus_client stub if the dependency is missing
if "prometheus_client" not in sys.modules:
    class _DummyCollector:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):
            return None

        def inc(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    fake_prom = types.ModuleType("prometheus_client")
    fake_prom.Counter = _DummyCollector
    fake_prom.Gauge = _DummyCollector
    fake_prom.Histogram = _DummyCollector
    fake_prom.Info = _DummyCollector
    fake_prom.REGISTRY = object()
    fake_prom.generate_latest = lambda registry=None: b""
    sys.modules["prometheus_client"] = fake_prom

monitoring_module = importlib.import_module("src.monitoring")
if not hasattr(monitoring_module, "get_prometheus_exporter"):
    monitoring_module.get_prometheus_exporter = monitoring_module.setup_prometheus

from src.api.server import app

client = TestClient(app)


def test_thrml_energy_endpoint_available():
    response = client.get("/thrml/energy")
    assert response.status_code == 200
    data = response.json()
    assert "energy" in data
    assert "timestamp" in data


def test_sampler_benchmarks_endpoint_available():
    response = client.get("/sampler/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "samples_per_sec" in data


def test_processor_list_endpoint_available():
    response = client.get("/processor/list")
    assert response.status_code == 200
    data = response.json()
    assert "processors" in data
    assert isinstance(data["processors"], list)
