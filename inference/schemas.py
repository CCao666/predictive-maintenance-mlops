"""HTTP request and response models."""

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    engine_id: int
    sequence: list[SensorReading] = Field(min_length=50, max_length=50)


class PredictionResponse(BaseModel):
    engine_id: int
    predicted_rul: float
    model_name: str
    model_version: str
    model_alias: str


class StatusResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str
    run_id: str
    sequence_length: int
    feature_count: int

