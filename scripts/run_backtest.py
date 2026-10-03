"""Step 2 - rolling-origin backtest of all forecasting models + prediction-interval coverage."""
import time

import numpy as np
import pandas as pd
from _common import FIGS, FIRST_ORIGIN, H, PROCESSED, REPORTS, SIM_END, SIM_START, load_processed

from invopt import figures
from invopt.forecast import ErrorLibrary, accuracy, interval_coverage, rolling_forecasts


def main():
    weekly, master = load_processed()
    Y = weekly.to_numpy()
    t = time.time()
    bt = rolling_forecasts(Y, weekly.index, range(FIRST_ORIGIN, SIM_END), H=H, skus=list(weekly.columns))
    print(f"backtest: {time.time() - t:.0f}s")
    bt.to_parquet(PROCESSED / "backtest.parquet")

    full = bt[bt["origin"] + H <= SIM_END]                       # origins with all 6 horizons observed
    pre = bt[bt["origin"] + H <= SIM_START - 1]                  # used to pick the model for the simulation
    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    acc = accuracy(full).sort_values("WAPE")
    acc_pre = accuracy(pre).sort_values("WAPE")
    acc_h = accuracy(full, by=("model", "h"))["WAPE"].unstack()
    best = acc.index[0]
    cov = pd.concat({m: interval_coverage(bt, m, H, range(SIM_START - 1, SIM_END - H + 1))["coverage_P10_P90"]
                     for m in ["ETS", "Ensemble"]}, axis=1)
    acc.round(4).to_csv(REPORTS / "forecast_accuracy.csv")
    acc_pre.round(4).to_csv(REPORTS / "forecast_accuracy_pre_simulation.csv")
    acc_h.round(4).to_csv(REPORTS / "forecast_wape_by_horizon.csv")
    cov.round(3).to_csv(REPORTS / "interval_coverage.csv")
    print("\nAll origins:\n", acc.round(3))
    print("\nPre-simulation origins (model selection):\n", acc_pre.round(3))
    print("\nWAPE by horizon:\n", acc_h.round(3))
    print("\nP10-P90 coverage (target 80%):\n", cov.round(3))

    figures.forecast_accuracy_bars(acc, FIGS / "forecast_accuracy.png",
                                   f"30 SKUs x {full['origin'].nunique()} weekly forecast origins "
                                   f"({weekly.index[FIRST_ORIGIN].date():%b %Y} - {weekly.index[SIM_END - H].date():%b %Y})")
    # example SKU: 1-week-ahead forecasts over the simulation period with the 80% band
    model = acc_pre.index[0]
    lib = ErrorLibrary(bt, model, H)
    sku = "85123A"
    d = bt[(bt["model"] == model) & (bt["h"] == 1) & (bt["sku"] == sku) & bt["origin"].between(SIM_START - 1, SIM_END - 1)]
    q = np.array([lib.quantiles(o, (0.1, 0.9))[:, 0] for o in d["origin"]])
    figures.forecast_example(d["week"], d["actual"], d["forecast"], d["forecast"] * q[:, 0], d["forecast"] * q[:, 1],
                             f"{sku} {master.loc[sku, 'description'].strip().title()} - {model} forecast",
                             FIGS / "forecast_example.png")


if __name__ == "__main__":
    main()
