"""Generate offline EDA and expanding-window diagnostics without opening final test."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from market_risk.data import features, validate_prices
from market_risk.research import walk_forward

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/SPY.csv")
p.add_argument("--output", default="artifacts/eda")
p.add_argument("--epochs", type=int, default=15)
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
df = validate_prices(pd.read_csv(a.csv, index_col=0, parse_dates=True))
f = features(df)
figures = []


def series(title, series):
    fig = go.Figure()
    for name, values in series.items():
        fig.add_scatter(x=values.index, y=values, name=name)
    fig.update_layout(title=title, template="plotly_white")
    figures.append(fig)


series("Adjusted prices", {"Adj Close": df["Adj Close"]})
series("Daily log returns", {"return": f["return"]})
series(
    "Historical annualized volatility",
    {"5 sessions": f.vol5, "10 sessions": f.vol10, "20 sessions": f.vol20},
)
series("Historical drawdown", {"drawdown": f.drawdown})
for name, values in [
    ("Return distribution", f["return"].dropna()),
    ("Volatility distribution", f.vol20.dropna()),
]:
    figures.append(
        go.Figure(go.Histogram(x=values)).update_layout(
            title=name, template="plotly_white"
        )
    )
for name, values in [
    ("Return autocorrelation", f["return"]),
    ("Squared-return autocorrelation", f["return"] ** 2),
]:
    figures.append(
        go.Figure(
            go.Bar(
                x=list(range(1, 31)), y=[values.autocorr(lag) for lag in range(1, 31)]
            )
        ).update_layout(title=name, template="plotly_white")
    )
html = '<html><head><meta charset="utf-8"><title>ENSI Market Risk EDA</title></head><body><h1>Market Risk · Historical EDA</h1><p>Retrospective observations; no trading recommendation.</p>'
for i, fig in enumerate(figures):
    html += fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
(root / "eda.html").write_text(html + "</body></html>")
report = walk_forward(df, folds=3, epochs=a.epochs)
(root / "walk_forward.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "rows": len(df),
            "figures": len(figures),
            "walk_forward_folds": len(report["folds"]),
        }
    )
)
