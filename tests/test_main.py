import os
import sys

import pytest
from fastapi.testclient import TestClient

# Let the tests find main.py and model.joblib inside the app folder
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

from main import app  # noqa: E402

VALID_HOUSE = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


@pytest.fixture(scope="module")
def client():
    # "with" runs the lifespan function, so the model is loaded
    with TestClient(app) as c:
        yield c


# UT-01: health check reports the model is loaded
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


# UT-02: a valid request returns the expected prediction
def test_predict_valid(client):
    response = client.post("/predict", json=VALID_HOUSE)
    assert response.status_code == 200
    value = response.json()["predicted_median_house_value"]
    assert value == pytest.approx(4.1521, abs=0.001)


# UT-03: latitude outside California is rejected
def test_predict_invalid_latitude(client):
    bad = {**VALID_HOUSE, "Latitude": 100}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# UT-04: a missing field is rejected
def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID_HOUSE.items() if k != "MedInc"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# UT-05: text in a number field is rejected
def test_predict_wrong_type(client):
    bad = {**VALID_HOUSE, "MedInc": "abc"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# UT-06: metrics endpoint exposes the request counter
def test_metrics(client):
    client.post("/predict", json=VALID_HOUSE)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert 'endpoint="/predict"' in response.text


# UT-07: an unknown path returns 404
def test_unknown_endpoint(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404