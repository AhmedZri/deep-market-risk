import json
import logging
import os
import secrets
import time
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .models import VolatilityLSTM, VolatilityCNN


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    sequence: list[list[float]] = Field(min_length=5, max_length=252)


def create_app(artifacts=None, token=None):
    root = Path(artifacts or os.getenv("ARTIFACT_DIR", "artifacts/spy"))
    key = token if token is not None else os.getenv("API_TOKEN", "")
    app = FastAPI(title="Deep Financial Market Risk", version="0.1.0")
    log = logging.getLogger("market_risk.api")

    def auth(x_api_key: str | None = Header(default=None)):
        if not key:
            raise HTTPException(
                503, "Configure API_TOKEN before using private endpoints"
            )
        if not x_api_key or not secrets.compare_digest(x_api_key, key):
            raise HTTPException(401, "Invalid API key")

    def metrics():
        path = root / "volatility_metrics.json"
        if not path.exists():
            raise HTTPException(503, "Train the model first")
        return json.loads(path.read_text())

    @app.exception_handler(RequestValidationError)
    async def validation_failure(request, exc):
        # Do not echo invalid values (NaN, infinities or personal data) into JSON errors.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in exc.errors()
                ]
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": (root / "volatility.pt").exists()}

    @app.get("/metrics", dependencies=[Depends(auth)])
    def get_metrics():
        return metrics()

    @app.post("/predict", dependencies=[Depends(auth)])
    def forecast(body: ForecastRequest):
        started = time.perf_counter()
        p = root / "volatility.pt"
        if not p.exists():
            raise HTTPException(503, "Train the model first")
        bundle = torch.load(p, map_location="cpu", weights_only=True)
        try:
            x = np.asarray(body.sequence, dtype=np.float32)
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(
                422, "A rectangular finite feature matrix is required"
            ) from exc
        if (
            x.shape != (bundle["lookback"], bundle["n_features"])
            or not np.isfinite(x).all()
        ):
            raise HTTPException(
                422,
                f"Expected finite {bundle['lookback']} x {bundle['n_features']} sequence in documented feature order",
            )
        model = {"lstm": VolatilityLSTM, "cnn": VolatilityCNN}[bundle["architecture"]](
            bundle["n_features"]
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        z = (x - np.asarray(bundle["mean"])) / np.asarray(bundle["scale"])
        with torch.no_grad():
            value = float(model(torch.tensor(z[None], dtype=torch.float32)).item())
        if not np.isfinite(value):
            raise HTTPException(422, "Input outside finite model inference domain")
        log.info(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "model": bundle["architecture"],
                    "duration_ms": 1000 * (time.perf_counter() - started),
                }
            )
        )
        return {
            "annualized_volatility": value,
            "horizon_sessions": bundle["horizon"],
            "model": bundle["architecture"],
            "source": bundle["source"],
            "feature_order": bundle["features"],
            "note": "Research estimate; no trading recommendation. The served neural model may differ from the best baseline.",
        }

    return app


app = create_app()
