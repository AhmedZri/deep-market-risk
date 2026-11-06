"""Point-in-time features and purged chronological sequences."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FEATURES = [
    "return",
    "vol5",
    "vol10",
    "vol20",
    "momentum",
    "range",
    "drawdown",
    "volume_change",
]


def validate_prices(df):
    if (
        not isinstance(df.index, pd.DatetimeIndex)
        or df.index.has_duplicates
        or not df.index.is_monotonic_increasing
    ):
        raise ValueError("A unique, increasing DatetimeIndex is required")
    if not set(COLUMNS) <= set(df.columns) or len(df) < 60:
        raise ValueError("OHLCV/Adj Close schema or history insufficient")
    a = df[COLUMNS].to_numpy(dtype=float)
    if (
        not np.isfinite(a).all()
        or (df[COLUMNS[:-1]] <= 0).any().any()
        or (df.Volume < 0).any()
    ):
        raise ValueError("Prices must be finite and positive; volume nonnegative")
    if (df.High < df[["Open", "Close", "Low"]].max(axis=1)).any() or (
        df.Low > df[["Open", "Close", "High"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC bounds")
    return df[COLUMNS].copy()


def synthetic_prices(n=600, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    op = close * np.exp(rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "Open": op,
            "High": np.maximum(op, close) * 1.005,
            "Low": np.minimum(op, close) * 0.995,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(10000, 50000, n),
        },
        index=pd.bdate_range("2015-01-01", periods=n, name="Date"),
    )


def features(df):
    df = validate_prices(df)
    p = df["Adj Close"]
    r = np.log(p).diff()
    return pd.DataFrame(
        {
            "return": r,
            "vol5": r.rolling(5).std() * np.sqrt(252),
            "vol10": r.rolling(10).std() * np.sqrt(252),
            "vol20": r.rolling(20).std() * np.sqrt(252),
            "momentum": p.pct_change(20),
            "range": (df.High - df.Low) / df.Close,
            "drawdown": p / p.cummax() - 1,
            "volume_change": np.log1p(df.Volume).diff(),
        },
        index=df.index,
    )


def sequences(df, lookback=20, horizon=5):
    if lookback < 2 or horizon < 2:
        raise ValueError("lookback and horizon must be >=2")
    f = features(df).dropna()
    r = np.log(df["Adj Close"]).diff()
    xs = []
    ys = []
    dates = []
    ends = []
    for j in range(lookback - 1, len(f)):
        date = f.index[j]
        i = df.index.get_loc(date)
        if i + horizon >= len(df):
            break
        xs.append(f.iloc[j - lookback + 1 : j + 1].to_numpy())
        ys.append(r.iloc[i + 1 : i + horizon + 1].std(ddof=1) * np.sqrt(252))
        dates.append(date)
        ends.append(df.index[i + horizon])
    if len(xs) < 40:
        raise ValueError("Not enough sequences for train/validation/test")
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        pd.DatetimeIndex(dates),
        pd.DatetimeIndex(ends),
    )


def split_indices(dates, ends):
    n = len(dates)
    a = int(n * 0.7)
    b = int(n * 0.85)
    tr = np.flatnonzero((np.arange(n) < a) & (ends < dates[a]))
    va = np.flatnonzero((np.arange(n) >= a) & (np.arange(n) < b) & (ends < dates[b]))
    te = np.arange(b, n)
    if min(len(tr), len(va), len(te)) < 2:
        raise ValueError("Insufficient purged split")
    return tr, va, te


def download(ticker, output):
    if ticker not in ["SPY", "^IXIC"]:
        raise ValueError("Only SPY and ^IXIC are in protocol")
    import yfinance as yf

    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2025-01-01",
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo unavailable; no synthetic substitution performed")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = validate_prices(df)
    if df.index.min() < pd.Timestamp("2015-01-01") or df.index.max() > pd.Timestamp(
        "2024-12-31"
    ):
        raise ValueError("Out-of-protocol dates")
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar("NYSE" if ticker == "SPY" else "NASDAQ")
    expected = calendar.valid_days(
        start_date="2015-01-01", end_date="2024-12-31"
    ).tz_localize(None)
    missing = expected.difference(df.index)
    unexpected = df.index.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Incomplete history/session coverage: missing={len(missing)}, unexpected={len(unexpected)}; first missing={list(missing[:5])}"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)
    manifest = {
        "source": "Yahoo Finance",
        "ticker": ticker,
        "start": "2015-01-01",
        "end": "2024-12-31",
        "rows": len(df),
        "actual_start": df.index.min().date().isoformat(),
        "actual_end": df.index.max().date().isoformat(),
        "expected_sessions": len(expected),
        "missing_sessions": 0,
        "calendar": "NYSE" if ticker == "SPY" else "NASDAQ",
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "price_convention": "Raw OHLCV and Yahoo Adj Close; return features use Adj Close",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    df.to_parquet(output.with_suffix(".parquet"))
    return manifest
