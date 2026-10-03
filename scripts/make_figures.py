"""Re-draw the README figures from saved results (no re-computation)."""
import numpy as np
import pandas as pd
from _common import FIGS, FIRST_ORIGIN, H, PROCESSED, REPORTS, SIM_END, load_processed

from invopt import figures


def main():
    weekly, master = load_processed()
    acc = pd.read_csv(REPORTS / "forecast_accuracy.csv", index_col=0)
    bt = pd.read_parquet(PROCESSED / "backtest.parquet")
    n_orig = bt[bt["origin"] + H <= SIM_END]["origin"].nunique()
    figures.forecast_accuracy_bars(acc, FIGS / "forecast_accuracy.png",
                                   f"30 SKUs x {n_orig} weekly forecast origins "
                                   f"({weekly.index[FIRST_ORIGIN]:%b %Y} - {weekly.index[SIM_END - H]:%b %Y})")
    summary = pd.read_csv(REPORTS / "policy_comparison.csv", index_col=0)
    figures.policy_cost_breakdown(summary[[c for c in summary.columns if "hindsight" not in c]],
                                  FIGS / "policy_cost_comparison.png")
    logs = pd.read_parquet(PROCESSED / "simulation_logs.parquet")
    logs = {k: g.droplevel(0) if isinstance(g.index, pd.MultiIndex) else g for k, g in logs.groupby(level=0)}
    order = ["Rule of thumb", "Optimised: point forecast + safety stock", "Optimised: scenarios (uncertainty-aware)"]
    sku = "85123A"
    figures.inventory_trajectory({k: logs[k] for k in order}, sku, master.loc[sku, "description"].strip(),
                                 FIGS / "inventory_trajectory.png")


if __name__ == "__main__":
    main()
