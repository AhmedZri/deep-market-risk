"""Reproducible research. Selection never reads the frozen test block."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from .data import sequences, split_indices, FEATURES, features
from .models import (
    VolatilityLSTM,
    VolatilityCNN,
    Autoencoder,
    RegimeLSTM,
    seed_all,
    fit_model,
)


def errors(y, p):
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(abs(y - p))),
    }


def infer(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()


def save_json(path, result):
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False))


def run_volatility(
    df,
    output,
    epochs=30,
    seeds=(42, 7, 123),
    lookback=20,
    source="Yahoo Finance",
    evaluate_test=False,
):
    if epochs < 1 or lookback < 5 or not seeds:
        raise ValueError("epochs >=1, lookback >=5, nonempty seeds required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    seal = out / "TEST_EVALUATED.json"
    if seal.exists():
        raise FileExistsError(
            "Test already opened in this experiment directory; keep frozen results"
        )
    x, y, dates, ends = sequences(df, lookback, 5)
    tr, va, te = split_indices(dates, ends)
    scaler = StandardScaler().fit(x[tr].reshape(-1, x.shape[-1]))
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
    f = features(df).loc[dates]
    baselines = {f"historical{w}": f[f"vol{w}"].to_numpy() for w in (5, 10, 20)}
    baselines["ewma"] = np.log(df["Adj Close"]).diff().ewm(
        alpha=0.06, adjust=False
    ).std().loc[dates].to_numpy() * np.sqrt(252)
    result = {
        "source": source,
        "horizon": 5,
        "lookback": lookback,
        "features": FEATURES,
        "protocol": "70/15/15 chronological; future labels purged; train-only scaler",
        "dataset_sha256": hashlib.sha256(df.to_csv().encode()).hexdigest(),
        "validation": {k: errors(y[va], p[va]) for k, p in baselines.items()},
        "splits": {
            k: {
                "n": len(idx),
                "start": str(dates[idx[0]].date()),
                "end": str(dates[idx[-1]].date()),
                "last_target": str(ends[idx[-1]].date()),
            }
            for k, idx in zip(["train", "validation", "test"], [tr, va, te])
        },
        "experiments": [],
    }
    models = {}
    for name, cls in [("lstm", VolatilityLSTM), ("cnn", VolatilityCNN)]:
        best = None
        scores = []
        for seed in seeds:
            seed_all(seed)
            model = cls(x.shape[-1])
            history = fit_model(model, z[tr], y[tr], z[va], y[va], epochs)
            score = errors(y[va], infer(model, z[va]))
            scores.append(score["rmse"])
            result["experiments"].append(
                {
                    "model": name,
                    "seed": seed,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "history": history,
                    "validation": score,
                }
            )
            if best is None or score["rmse"] < best[0]:
                best = (score["rmse"], model, seed, score)
        models[name] = best[1]
        result["validation"][name] = best[3]
        result["validation"][name]["seed_rmse_std"] = float(np.std(scores))
    selected = min(result["validation"], key=lambda k: result["validation"][k]["rmse"])
    # The neural artifact also remains available if a baseline wins.
    neural = min(models, key=lambda k: result["validation"][k]["rmse"])
    artifact = {
        "state_dict": models[neural].state_dict(),
        "architecture": neural,
        "n_features": x.shape[-1],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "lookback": lookback,
        "horizon": 5,
        "source": source,
        "features": FEATURES,
    }
    torch.save(artifact, out / "volatility.pt")
    # Expanding causal baseline diagnostics, entirely within train+validation.
    result["walk_forward"] = []
    for stop in np.linspace(max(30, len(tr) // 2), int(0.85 * len(x)), 4, dtype=int)[
        1:
    ]:
        begin = max(0, stop - 30)
        result["walk_forward"].append(
            {
                "start": str(dates[begin]),
                "end": str(dates[stop - 1]),
                "historical20": errors(
                    y[begin:stop], baselines["historical20"][begin:stop]
                ),
            }
        )
    idx = va
    result["selected_model"] = selected
    result["test"] = None
    if evaluate_test:
        # O_EXCL claim before accessing test metrics prevents accidental repeated final evaluation.
        with seal.open("x") as handle:
            json.dump(
                {"dataset": result["dataset_sha256"], "selected": selected}, handle
            )
        pred = (
            infer(models[selected], z[te])
            if selected in models
            else baselines[selected][te]
        )
        result["test"] = {
            "selected_model": selected,
            **errors(y[te], pred),
            "baselines": {
                name: errors(y[te], values[te]) for name, values in baselines.items()
            },
        }
        idx = te
    predictions = pd.DataFrame(
        {
            "date": dates[idx],
            "actual": y[idx],
            **{k: p[idx] for k, p in baselines.items()},
            **{k: infer(m, z[idx]) for k, m in models.items()},
        }
    )
    predictions.to_csv(out / "volatility_predictions.csv", index=False)
    residual = abs(y[va] - infer(models[neural], z[va]))
    result["validation_absolute_error_q90"] = float(np.quantile(residual, 0.9))
    result["limitations"] = [
        "Adjusted prices are a retrospective Yahoo snapshot, not point-in-time corporate-action vintages.",
        "Error quantile is descriptive validation uncertainty, not a coverage guarantee.",
        "Synthetic results do not validate Yahoo or financial forecasting.",
    ]
    save_json(out / "volatility_metrics.json", result)
    return result


def run_regimes(
    df, output, epochs=30, source="Yahoo Finance", seed=42, evaluate_test=False
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    seal = out / "REGIMES_TEST_EVALUATED.json"
    if seal.exists():
        raise FileExistsError(
            "Regime pseudo-label test already evaluated in this experiment directory"
        )
    x, _, dates, ends = sequences(df, 20, 5)
    tr, va, te = split_indices(dates, ends)
    scaler = StandardScaler().fit(x[tr, -1])
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
    ae = Autoencoder(x.shape[-1])
    fit_model(ae, z[tr, -1], z[tr, -1], z[va, -1], z[va, -1], epochs, autoencoder=True)
    with torch.no_grad():
        latent = ae.encode(torch.tensor(z[:, -1])).numpy()
    km = KMeans(3, random_state=seed, n_init=10).fit(latent[tr])
    labels = km.predict(latent)
    classifier = RegimeLSTM(x.shape[-1], 3)
    fit_model(
        classifier, z[tr], labels[tr], z[va], labels[va], epochs, classification=True
    )
    evaluation = te if evaluate_test else va
    if evaluate_test:
        with seal.open("x") as f:
            json.dump(
                {
                    "seed": seed,
                    "source": source,
                    "scope": "Frozen pseudo-label concordance evaluation",
                },
                f,
            )
    pred = infer(classifier, z[evaluation]).argmax(axis=1)
    profiles = (
        features(df)
        .loc[dates]
        .assign(cluster=labels)
        .iloc[tr]
        .groupby("cluster")
        .mean()
        .reset_index()
        .to_dict("records")
    )
    stability = []
    rng = np.random.default_rng(seed)
    for _ in range(5):
        bootstrap = rng.choice(tr, len(tr), replace=True)
        other = KMeans(3, random_state=seed, n_init=10).fit(latent[bootstrap])
        stability.append(adjusted_rand_score(labels[va], other.predict(latent[va])))
    transitions = pd.crosstab(
        pd.Series(labels[tr][:-1], name="from"),
        pd.Series(labels[tr][1:], name="to"),
        normalize="index",
    ).reindex(index=range(3), columns=range(3), fill_value=0)
    result = {
        "source": source,
        "label_semantics": "train-only latent cluster pseudo-labels; not economic truth",
        "profiles": profiles,
        "stability_ari": float(np.mean(stability)),
        "stability_scope": "KMeans bootstrap in fixed train-fitted autoencoder latent space; not full network stability",
        "evaluation_split": "test" if evaluate_test else "validation",
        "pseudo_label_report": classification_report(
            labels[evaluation], pred, output_dict=True, zero_division=0
        ),
        "pseudo_label_baseline": classification_report(
            labels[evaluation],
            np.full(len(evaluation), np.bincount(labels[tr]).argmax()),
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_matrix(
            labels[evaluation], pred, labels=[0, 1, 2]
        ).tolist(),
        "transitions": transitions.to_numpy().tolist(),
        "seed": seed,
        "features": FEATURES,
        "dataset_sha256": hashlib.sha256(df.to_csv().encode()).hexdigest(),
    }
    import joblib

    joblib.dump({"scaler": scaler, "kmeans": km}, out / "regime_preprocessing.joblib")
    torch.save(
        {
            "autoencoder": ae.state_dict(),
            "lstm": classifier.state_dict(),
            "features": FEATURES,
        },
        out / "regimes.pt",
    )
    pd.DataFrame({"date": dates, "statistical_regime": labels}).to_csv(
        out / "regimes.csv", index=False
    )
    save_json(out / "regime_metrics.json", result)
    return result
