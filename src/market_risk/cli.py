import os
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import download, synthetic_prices
from .pipeline import run_volatility, run_regimes


def main():
    parser = argparse.ArgumentParser(description="Causal daily market-risk research")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--ticker", choices=["SPY", "^IXIC"], required=True)
    d.add_argument("--output", required=True)
    for name in ["volatility", "regimes", "demo"]:
        p = sub.add_parser(name)
        p.add_argument("--csv")
        p.add_argument("--output", required=True)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--evaluate-test", action="store_true")
        p.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(args.ticker, args.output), indent=2))
        return
    if args.command != "demo" and not args.csv:
        parser.error("--csv required for real-data experiments")
    source = "synthetic" if args.command == "demo" else "Yahoo Finance"
    if args.command == "demo":
        df = synthetic_prices()
    else:
        path = Path(args.csv)
        manifest = path.with_suffix(".manifest.json")
        if not manifest.exists():
            parser.error("Source manifest missing: download with this CLI first")
        import hashlib

        info = json.loads(manifest.read_text())
        expected = "SPY" if args.command == "volatility" else "^IXIC"
        if (
            info["ticker"] != expected
            or info["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            parser.error("Ticker or snapshot checksum mismatch")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    if args.command == "regimes":
        result = run_regimes(
            df,
            args.output,
            epochs=args.epochs,
            source=source,
            evaluate_test=args.evaluate_test,
        )
    else:
        result = run_volatility(
            df,
            args.output,
            epochs=args.epochs,
            source=source,
            evaluate_test=args.evaluate_test,
        )
    if args.mlflow:
        import mlflow

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
        mlflow.set_experiment("market-risk")
        with mlflow.start_run():
            mlflow.log_params(
                {"source": source, "epochs": args.epochs, "module": args.command}
            )
            mlflow.log_artifacts(args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["experiments"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
