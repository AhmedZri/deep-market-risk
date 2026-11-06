import numpy as np
import pytest
import torch
from market_risk.models import (
    VolatilityLSTM,
    VolatilityCNN,
    Autoencoder,
    RegimeLSTM,
    fit_model,
)
from market_risk.pipeline import run_volatility, run_regimes
from market_risk.data import synthetic_prices


@pytest.mark.parametrize("cls", [VolatilityLSTM, VolatilityCNN])
def test_positive_output_and_roundtrip(cls, tmp_path):
    torch.manual_seed(42)
    model = cls(8)
    x = torch.randn(4, 20, 8)
    out = model(x)
    assert out.shape == (4,)
    assert (out > 0).all()
    assert torch.isfinite(out).all()
    p = tmp_path / "model.pt"
    torch.save(model.state_dict(), p)
    other = cls(8)
    other.load_state_dict(torch.load(p, weights_only=True))
    torch.testing.assert_close(out, other(x))


def test_regime_architectures():
    a = Autoencoder(8)
    x = torch.randn(6, 8)
    assert a(x).shape == x.shape
    assert a.encode(x).shape == (6, 3)
    assert RegimeLSTM(8, 3)(torch.randn(6, 12, 8)).shape == (6, 3)


def test_volatility_smoke_and_test_seal(tmp_path):
    result = run_volatility(
        synthetic_prices(250),
        tmp_path,
        epochs=1,
        seeds=(42,),
        lookback=10,
        source="synthetic",
        evaluate_test=True,
    )
    assert set(result["validation"]) == {
        "lstm",
        "cnn",
        "historical5",
        "historical10",
        "historical20",
        "ewma",
    }
    assert result["test"]["selected_model"]
    assert (tmp_path / "volatility.pt").exists()
    with pytest.raises(FileExistsError):
        run_volatility(
            synthetic_prices(250),
            tmp_path,
            epochs=1,
            seeds=(42,),
            lookback=10,
            source="synthetic",
            evaluate_test=True,
        )


def test_regime_smoke(tmp_path):
    result = run_regimes(synthetic_prices(250), tmp_path, epochs=1, source="synthetic")
    assert len(result["profiles"]) == 3
    assert (
        result["label_semantics"]
        == "train-only latent cluster pseudo-labels; not economic truth"
    )
    assert np.isfinite(result["stability_ari"])


def test_regime_test_is_opt_in_and_sealed(tmp_path):
    prices = synthetic_prices(240)
    validation = run_regimes(prices, tmp_path, epochs=1, source="synthetic")
    assert validation["evaluation_split"] == "validation"
    assert not (tmp_path / "REGIMES_TEST_EVALUATED.json").exists()
    final = run_regimes(
        prices, tmp_path, epochs=1, source="synthetic", evaluate_test=True
    )
    assert final["evaluation_split"] == "test"
    with pytest.raises(FileExistsError):
        run_regimes(prices, tmp_path, epochs=1, source="synthetic", evaluate_test=True)
