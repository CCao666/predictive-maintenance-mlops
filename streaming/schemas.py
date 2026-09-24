"""Messages exchanged through the sensor topic."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from inference.schemas import SensorReading


DRIFT_TOPIC = "model-drift-events"


class SensorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(default="FD001", pattern=r"^FD00[1-4]$")
    engine_id: int = Field(gt=0)
    time_cycle: int = Field(gt=0)
    features: SensorReading


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(default="FD001", pattern=r"^FD00[1-4]$")
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


class DriftEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^FD00[1-4]$")
    severity: Literal["none", "warning", "critical"]
    drift_detected: bool
    drift_score: float = Field(ge=0, le=1)
    drifted_feature_ratio: float = Field(ge=0, le=1)
    drifted_features: list[str]
    reference_rows: int = Field(gt=0)
    current_rows: int = Field(gt=0)
    observed_at: datetime = Field(
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
