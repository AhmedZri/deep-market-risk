"""Expanding-window diagnostics restricted to train+validation, never final test."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from .data import sequences, features
from .models import seed_all, VolatilityLSTM, VolatilityCNN, fit_model
from .pipeline import infer, errors


def walk_forward(df, folds=3, epochs=15, seed=42):
    x, y, dates, ends = sequences(df, 20, 5)
    test_start = dates[int(0.85 * len(x))]
    allowed = np.flatnonzero(ends < test_start)
    reports = []
    for tr, va in TimeSeriesSplit(n_splits=folds, gap=5).split(allowed):
        tr = allowed[tr]
        va = allowed[va]
        tr = tr[ends[tr] < dates[va[0]]]
        scaler = StandardScaler().fit(x[tr].reshape(-1, x.shape[-1]))
        z = (
            scaler.transform(x.reshape(-1, x.shape[-1]))
            .reshape(x.shape)
            .astype("float32")
        )
        metrics = {}
        for name, cls in [("lstm", VolatilityLSTM), ("cnn", VolatilityCNN)]:
            seed_all(seed)
            model = cls(x.shape[-1])
            fit_model(model, z[tr], y[tr], z[va], y[va], epochs)
            metrics[name] = errors(y[va], infer(model, z[va]))
        metrics["historical20"] = errors(
            y[va], features(df).loc[dates[va], "vol20"].to_numpy()
        )
        reports.append(
            {
                "last_train_target": str(ends[tr].max()),
                "validation_start": str(dates[va].min()),
                "last_validation_target": str(ends[va].max()),
                "metrics": metrics,
            }
        )
    return {
        "frozen_test_start": str(test_start),
        "seed": seed,
        "folds": reports,
        "scope": "Expanding validation diagnostics with early stopping; not an independent final test.",
    }
