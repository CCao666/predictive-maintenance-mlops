"""Per-engine sliding windows used for online inference."""

from collections import defaultdict, deque

from streaming.schemas import SensorEvent


class EngineWindowStore:
    def __init__(self, sequence_length: int = 50) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")

        self.sequence_length = sequence_length
        self._windows: dict[int, deque[dict]] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )
        self._last_cycle: dict[int, int] = {}

    def add(self, event: SensorEvent) -> list[dict] | None:
        """Add one reading and return a full window when it is ready."""
        last_cycle = self._last_cycle.get(event.engine_id, 0)
        if event.time_cycle <= last_cycle:
            raise ValueError(
                f"Engine {event.engine_id} received cycle {event.time_cycle} "
                f"after cycle {last_cycle}"
            )

        self._last_cycle[event.engine_id] = event.time_cycle
        window = self._windows[event.engine_id]
        window.append(event.features.model_dump())

        if len(window) < self.sequence_length:
            return None
        return list(window)
