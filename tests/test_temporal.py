import numpy as np
import pandas as pd
import pytest
from market_risk.data import (
    features,
    sequences,
    split_indices,
    validate_prices,
    synthetic_prices,
)


@pytest.fixture
def prices():
    return synthetic_prices(400, 42)


@pytest.mark.parametrize("cut", [80, 100, 150, 200, 250, 300, 350, 380, 390, 395])
def test_features_do_not_look_ahead(prices, cut):
    a = features(prices)
    b = prices.copy()
    b.iloc[cut:, b.columns.get_loc("Adj Close")] *= 4
    pd.testing.assert_frame_equal(a.iloc[:cut], features(b).iloc[:cut])


def test_future_target_exact(prices):
    x, y, dates, ends = sequences(prices, lookback=12, horizon=5)
    i = prices.index.get_loc(dates[0])
    r = np.log(prices["Adj Close"]).diff()
    assert y[0] == pytest.approx(r.iloc[i + 1 : i + 6].std(ddof=1) * np.sqrt(252))
    assert ends[0] == prices.index[i + 5]
    np.testing.assert_allclose(x[0, -1], features(prices).loc[dates[0]])


def test_split_purges_future_labels(prices):
    x, y, d, e = sequences(prices, 12, 5)
    tr, va, te = split_indices(d, e)
    assert e[tr].max() < d[va].min()
    assert e[va].max() < d[te].min()
    assert len(set(tr) & set(va)) == 0


@pytest.mark.parametrize(
    "kind", ["duplicate", "negative", "missing", "unsorted", "infinite"]
)
def test_invalid_prices_rejected(prices, kind):
    p = prices.copy()
    if kind == "duplicate":
        p.index = pd.Index([p.index[0]] * len(p))
    if kind == "negative":
        p.iloc[1, p.columns.get_loc("Adj Close")] = -1
    if kind == "missing":
        p = p.drop(columns="Volume")
    if kind == "unsorted":
        p = p.iloc[::-1]
    if kind == "infinite":
        p.iloc[1, 0] = np.inf
    with pytest.raises(ValueError):
        validate_prices(p)


def test_download_rejects_partial_history(monkeypatch, tmp_path):
    import yfinance
    from market_risk.data import download

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: synthetic_prices(220))
    with pytest.raises(ValueError, match="coverage|session|history"):
        download("SPY", tmp_path / "partial.csv")
