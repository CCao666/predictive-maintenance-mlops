import httpx
import pytest

from streaming.consumer import build_prediction_event, request_prediction
from streaming.producer import infer_dataset_id, iter_events
from streaming.schemas import SensorEvent, classify_rul
from streaming.window import EngineWindowStore


FEATURES = {
    "op_setting_1": 0.0,
    "op_setting_2": 0.0,
    "op_setting_3": 100.0,
    **{f"sensor_{index}": float(index) for index in range(1, 22)},
}


def event(engine_id: int, cycle: int) -> SensorEvent:
    return SensorEvent(
        engine_id=engine_id,
        time_cycle=cycle,
        features=FEATURES,
    )


def test_windows_are_kept_separate_by_engine():
    store = EngineWindowStore(sequence_length=2)

    assert store.add(event(1, 1)) is None
    assert store.add(event(2, 1)) is None
    assert len(store.add(event(1, 2))) == 2
    assert len(store.add(event(2, 2))) == 2


def test_window_slides_after_it_is_full():
    store = EngineWindowStore(sequence_length=2)
    store.add(event(1, 1))
    first = store.add(event(1, 2))
    second = store.add(event(1, 3))

    assert first is not None
    assert second is not None
    assert len(second) == 2


def test_out_of_order_cycle_is_rejected():
    store = EngineWindowStore()
    store.add(event(1, 2))

    with pytest.raises(ValueError, match="received cycle 1 after cycle 2"):
        store.add(event(1, 1))


def test_iter_events_builds_valid_messages(tmp_path):
    row = "1 1 0.1 0.2 100 " + " ".join(str(i) for i in range(1, 22))
    path = tmp_path / "sample.txt"
    path.write_text(row + "\n")

    result = list(iter_events(path))

    assert len(result) == 1
    assert result[0].dataset_id == "FD001"
    assert result[0].engine_id == 1
    assert result[0].features.sensor_21 == 21


def test_dataset_id_is_inferred_from_filename():
    assert infer_dataset_id("data/cmapss/test_FD004.txt") == "FD004"


def test_prediction_request_uses_expected_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        assert body == {"engine_id": 7, "sequence": [FEATURES]}
        return httpx.Response(200, json={"predicted_rul": 12.5})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = request_prediction(client, "http://api", 7, [FEATURES])

    assert result["predicted_rul"] == 12.5


@pytest.mark.parametrize(
    ("rul", "expected"),
    [
        (10, ("imminent", "urgent")),
        (30, ("critical", "critical")),
        (45, ("warning", "warning")),
        (55, ("watch", "info")),
        (70, ("healthy", "none")),
    ],
)
def test_classify_rul(rul, expected):
    assert classify_rul(rul) == expected


def test_prediction_event_keeps_engine_cycle_and_model_metadata():
    prediction = build_prediction_event(
        event(7, 50),
        {
            "predicted_rul": 12.5,
            "model_name": "predictive-maintenance-rul",
            "model_version": "1",
            "model_alias": "champion",
        },
    )

    assert prediction.engine_id == 7
    assert prediction.dataset_id == "FD001"
    assert prediction.time_cycle == 50
    assert prediction.health_status == "imminent"
    assert prediction.alert_level == "urgent"


def test_dataset_windows_are_kept_separate():
    store = EngineWindowStore(sequence_length=2)
    fd001 = event(1, 1)
    fd002 = event(1, 1).model_copy(update={"dataset_id": "FD002"})

    assert store.add(fd001) is None
    assert store.add(fd002) is None
