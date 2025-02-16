import pytest
from fastapi.testclient import TestClient
from src.web.interface import WebServer
from src.neural.cortex import DefaultCortex

@pytest.fixture
def client():
    cortex = DefaultCortex()
    server = WebServer(cortex)
    return TestClient(server.app)

def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_ask_endpoint(client):
    response = client.post("/api/ask", json={"query": "test question"})
    assert response.status_code == 200
    assert isinstance(response.json(), str) 