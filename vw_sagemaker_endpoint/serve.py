"""FastAPI serving app for SageMaker real-time VW inference."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Response, status
from pydantic import BaseModel, Field

from predictor import VWPredictor


MODEL_PATH = "/opt/ml/model/pl_aip_uplift_model.vw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VW SageMaker Endpoint", version="1.0.0")
predictor: VWPredictor | None = None
startup_error: str | None = None


class InvocationPayload(BaseModel):
    use_case_id: str | None = None
    master_user_id: str | None = None
    context: Dict[str, Any] | None = Field(default=None)
    candidate_actions: list[Dict[str, Any]] | None = Field(default=None)


@app.on_event("startup")
def load_model() -> None:
    global predictor, startup_error
    try:
        logger.info("Loading VW model from %s", MODEL_PATH)
        predictor = VWPredictor(MODEL_PATH)
        startup_error = None
        logger.info("VW model loaded successfully")
    except Exception as exc:  # noqa: BLE001 - health endpoint must expose startup state.
        predictor = None
        startup_error = str(exc)
        logger.exception("Failed to load VW model")


@app.get("/ping")
def ping(response: Response) -> Dict[str, Any]:
    if predictor is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "FAILED", "error": startup_error or "model not loaded"}
    return {"status": "OK"}


@app.post("/invocations")
def invocations(payload: InvocationPayload) -> Dict[str, Any]:
    request_payload = payload.dict()

    if predictor is None:
        return _failed(request_payload, startup_error or "model not loaded")
    if not request_payload.get("context"):
        return _failed(request_payload, "context is required")
    if not request_payload.get("candidate_actions"):
        return _failed(request_payload, "candidate_actions must contain at least one action")

    try:
        return predictor.predict(request_payload)
    except Exception as exc:  # noqa: BLE001 - SageMaker should receive a JSON failure body.
        logger.exception("Inference failed")
        return _failed(request_payload, str(exc))


def _failed(payload: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {
        "decision_status": "FAILED",
        "use_case_id": payload.get("use_case_id"),
        "master_user_id": payload.get("master_user_id"),
        "model_type": "vowpal_wabbit_cb_adf",
        "recommendations": [],
        "debug": {
            "candidate_action_count": len(payload.get("candidate_actions") or []),
            "error": error,
        },
    }
