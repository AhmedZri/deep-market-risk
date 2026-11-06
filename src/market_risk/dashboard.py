import json
import os
from pathlib import Path
import secrets
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Market Risk Research", page_icon="📈", layout="wide")
st.title("Financial Market Risk · Research Lab")
st.caption(
    "SPY volatility and NASDAQ statistical regimes | Daily observations, 2015–2024"
)
key = os.getenv("API_TOKEN", "")
entered = st.sidebar.text_input("Local access token", type="password")
if not key or not secrets.compare_digest(entered, key):
    st.info("Set API_TOKEN, then enter it in the sidebar.")
    st.stop()
root = Path(st.sidebar.text_input("Artifacts root", "artifacts"))
page = st.sidebar.radio("Analysis", ["Volatility", "NASDAQ regimes", "Protocol"])
if page == "Protocol":
    st.markdown(
        "**Causal features → purged 70/15/15 split → train-only scaling → validation selection → sealed test.**"
    )
    st.warning(
        "Statistical regimes are pseudo-labels. Historical adjusted Yahoo prices are retrospective. No automated trading."
    )
    st.json(
        {
            "annualization": 252,
            "future_volatility_horizon": 5,
            "lookback": 20,
            "test_tuning": False,
        }
    )
    st.stop()
if page == "Volatility":
    p = root / "spy" / "volatility_metrics.json"
    if not p.exists():
        st.info("Run the volatility CLI first.")
        st.stop()
    m = json.loads(p.read_text())
    st.info("Data source: " + m["source"])
    a, b, c = st.columns(3)
    a.metric("Selected model", m["selected_model"])
    b.metric("Test RMSE", f"{m['test']['rmse']:.4f}" if m["test"] else "Sealed")
    c.metric("Horizon", "5 sessions")
    comparison = pd.DataFrame(m["validation"]).T.reset_index(names="model")
    st.plotly_chart(
        px.bar(
            comparison, x="model", y="rmse", title="Validation RMSE · lower is better"
        ),
        width="stretch",
    )
    df = pd.read_csv(root / "spy" / "volatility_predictions.csv", parse_dates=["date"])
    st.plotly_chart(
        px.line(
            df, x="date", y=["actual", "lstm", "cnn"], title="Annualized volatility"
        ),
        width="stretch",
    )
    residual = df.assign(
        error=df[m["selected_model"]] - df.actual, year=df.date.dt.year
    )
    st.plotly_chart(
        px.box(residual, x="year", y="error", title="Error distribution by year"),
        width="stretch",
    )
    st.plotly_chart(
        px.histogram(residual, x="error", title="Residual distribution"),
        width="stretch",
    )
    st.dataframe(comparison, width="stretch")
    st.download_button(
        "Export predictions CSV", df.to_csv(index=False), "predictions.csv"
    )
else:
    p = root / "nasdaq" / "regime_metrics.json"
    if not p.exists():
        st.info("Run the regimes CLI first.")
        st.stop()
    m = json.loads(p.read_text())
    df = pd.read_csv(root / "nasdaq" / "regimes.csv", parse_dates=["date"])
    st.warning(m["label_semantics"])
    st.metric("Bootstrap ARI · fixed encoder", f"{m['stability_ari']:.3f}")
    st.plotly_chart(
        px.scatter(
            df,
            x="date",
            y="statistical_regime",
            color=df.statistical_regime.astype(str),
            title="Estimated statistical states",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.imshow(
            m["transitions"],
            text_auto=".2f",
            title="Train-state transition probabilities",
        ),
        width="stretch",
    )
    st.dataframe(pd.DataFrame(m["profiles"]), width="stretch")
    st.json(m["pseudo_label_report"])
