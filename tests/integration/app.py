"""Deterministic model substitute for testing the real service pipeline."""

from inference.main import create_app


class TestModelManager:
    ready = False

    def load(self):
        self.ready = True

    def predict(self, sequence):
        # The final sensor value controls the RUL for deterministic alert tests.
        return float(sequence[-1].sensor_1)

    def info(self):
        return {
            "name": "ci-model",
            "version": "ci-1",
            "alias": "test",
            "run_id": "ci-run",
            "sequence_length": 50,
            "feature_count": 24,
        }


def create_test_app():
    return create_app(TestModelManager())
