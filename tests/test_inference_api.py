"""Tests for the FastAPI inference endpoints."""

from fastapi.testclient import TestClient

from inference.main import create_app
from training.dataset import FEATURE_COLUMNS


def sensor_reading() -> dict[str, float]:
    return {name: 1.0 for name in FEATURE_COLUMNS}


class FakeModelManager:
    ready = False

    def load(self) -> None:
        self.ready = True

    def predict(self, sequence: list) -> float:
        assert len(sequence) == 50
        return 42.5

    def info(self) -> dict:
        return {
            "name": "predictive-maintenance-rul",
            "version": "1",
            "alias": "champion",
            "run_id": "test-run",
            "sequence_length": 50,
            "feature_count": 24,
        }


class FailingModelManager(FakeModelManager):
    def load(self) -> None:
        raise RuntimeError("Registry unavailable")


def test_health_and_readiness():
    with TestClient(create_app(FakeModelManager())) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        assert client.get("/ready").json() == {"status": "ready"}


def test_readiness_fails_when_model_did_not_load():
    with TestClient(create_app(FailingModelManager())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_model_info():
    with TestClient(create_app(FakeModelManager())) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["version"] == "1"
    assert response.json()["feature_count"] == 24


def test_predict():
    request = {
        "engine_id": 7,
        "sequence": [sensor_reading() for _ in range(50)],
    }

    with TestClient(create_app(FakeModelManager())) as client:
        response = client.post("/predict", json=request)

    assert response.status_code == 200
    assert response.json() == {
        "engine_id": 7,
        "predicted_rul": 42.5,
        "model_name": "predictive-maintenance-rul",
        "model_version": "1",
        "model_alias": "champion",
    }


def test_predict_rejects_wrong_sequence_length():
    request = {
        "engine_id": 7,
        "sequence": [sensor_reading() for _ in range(49)],
    }

    with TestClient(create_app(FakeModelManager())) as client:
        response = client.post("/predict", json=request)

    assert response.status_code == 422


def test_metrics_are_exposed():
    with TestClient(create_app(FakeModelManager())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text

