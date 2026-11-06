from fastapi.testclient import TestClient
from market_risk.api import create_app


def test_fail_closed_and_missing_model(tmp_path):
    c = TestClient(create_app(tmp_path, token="secret"))
    assert c.get("/health").status_code == 200
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"X-API-Key": "secret"}).status_code == 503
    r = c.post(
        "/predict", headers={"X-API-Key": "secret"}, json={"sequence": [[1] * 8] * 5}
    )
    assert r.status_code == 503


def test_no_configured_token_fails_closed(tmp_path):
    c = TestClient(create_app(tmp_path, token=""))
    assert c.get("/metrics", headers={"X-API-Key": "anything"}).status_code == 503


def test_trained_api_matches_model_and_rejects_shape(tmp_path):
    import numpy as np
    import torch
    from market_risk.data import synthetic_prices, sequences
    from market_risk.pipeline import run_volatility
    from market_risk.models import VolatilityLSTM, VolatilityCNN

    prices = synthetic_prices(220)
    run_volatility(prices, tmp_path, epochs=1, seeds=(42,), source="synthetic")
    sequence = sequences(prices)[0][0]
    c = TestClient(create_app(tmp_path, token="secret"))
    h = {"X-API-Key": "secret"}
    r = c.post("/predict", headers=h, json={"sequence": sequence.tolist()})
    assert r.status_code == 200
    bundle = torch.load(tmp_path / "volatility.pt", weights_only=True)
    model = {"lstm": VolatilityLSTM, "cnn": VolatilityCNN}[bundle["architecture"]](8)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    z = (sequence - np.asarray(bundle["mean"])) / np.asarray(bundle["scale"])
    with torch.no_grad():
        expected = float(model(torch.tensor(z[None], dtype=torch.float32)).item())
    assert abs(r.json()["annualized_volatility"] - expected) < 1e-6
    assert (
        c.post("/predict", headers=h, json={"sequence": [[1.0] * 7] * 20}).status_code
        == 422
    )
    assert (
        c.post(
            "/predict",
            headers={**h, "Content-Type": "application/json"},
            content='{"sequence": [[NaN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0], [1.0], [1.0], [1.0]]}',
        ).status_code
        == 422
    )


def test_ragged_input_is_422(tmp_path):
    from market_risk.pipeline import run_volatility
    from market_risk.data import synthetic_prices

    run_volatility(
        synthetic_prices(220), tmp_path, epochs=1, seeds=(42,), source="synthetic"
    )
    c = TestClient(create_app(tmp_path, token="secret"))
    assert (
        c.post(
            "/predict",
            headers={"X-API-Key": "secret"},
            json={"sequence": [[1.0] * 8] * 19 + [[1.0] * 7]},
        ).status_code
        == 422
    )
