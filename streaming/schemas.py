"""Messages exchanged through the sensor topic."""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from inference.schemas import SensorReading


class SensorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(gt=0)
    time_cycle: int = Field(gt=0)
    features: SensorReading


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(gt=0)
    time_cycle: int = Field(gt=0)
    predicted_rul: float = Field(ge=0)
    health_status: str
    alert_level: str
    model_name: str
    model_version: str
    model_alias: str
    predicted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def classify_rul(rul: float) -> tuple[str, str]:
    """Return health status and the conservative production alert level."""
    if rul <= 15:
        return "imminent", "urgent"
    if rul <= 40:
        return "critical", "critical"
    if rul <= 50:
        return "warning", "warning"
    if rul <= 60:
        return "watch", "info"
    return "healthy", "none"
