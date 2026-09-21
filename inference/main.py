"""FastAPI application for online RUL prediction."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inference.metrics import (
    ENGINE_LAST_PREDICTION,
    ENGINE_RUL,
    MODEL_INFO,
    PREDICTED_RUL,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)
from inference.model_manager import ModelManager
from inference.schemas import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    StatusResponse,
)


logger = logging.getLogger(__name__)


def create_app(model_manager: ModelManager | None = None) -> FastAPI:
    manager = model_manager or ModelManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            manager.load()
            info = manager.info()
            MODEL_INFO.labels(
                info["name"],
                info["version"],
                info["alias"],
            ).set(1)
        except Exception:
            logger.exception("Model failed to load during startup")
        yield

    app = FastAPI(
        title="Predictive Maintenance Inference API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=StatusResponse)
    def health() -> StatusResponse:
        return StatusResponse(status="healthy")

    @app.get("/ready", response_model=StatusResponse)
    def ready() -> StatusResponse:
        if not manager.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return StatusResponse(status="ready")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> dict:
        if not manager.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return manager.info()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if not manager.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        PREDICTION_REQUESTS.inc()
        try:
            with PREDICTION_LATENCY.time():
                predicted_rul = manager.predict(request.sequence)
            PREDICTED_RUL.observe(predicted_rul)
            ENGINE_RUL.labels(engine_id=str(request.engine_id)).set(
                predicted_rul
            )
            ENGINE_LAST_PREDICTION.labels(
                engine_id=str(request.engine_id)
            ).set(time.time())
        except ValueError as error:
            PREDICTION_ERRORS.inc()
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            PREDICTION_ERRORS.inc()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from error

        info = manager.info()
        return PredictionResponse(
            engine_id=request.engine_id,
            predicted_rul=predicted_rul,
            model_name=info["name"],
            model_version=info["version"],
            model_alias=info["alias"],
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
