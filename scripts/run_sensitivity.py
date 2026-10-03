"""Sensitivity of the main result to the assumed fixed cost per order line.

    python scripts/run_sensitivity.py 5 30      # one simulation per value (~5-10 min each)
"""
import sys

import pandas as pd
from _common import PROCESSED, REPORTS, SIM_END, SIM_START, load_processed
from run_simulation import run_policies, setup

from invopt.params import Assumptions


def main(values):
    weekly, master = load_processed()
    bt = pd.read_parquet(PROCESSED / "backtest.parquet")
    for K in values:
        P, S, model = setup(weekly, master, bt, SIM_START, Assumptions(fixed_order_cost=K))
        sm, _ = run_policies(weekly.to_numpy(), weekly.index, P, S, bt, model, SIM_START, SIM_END, label=f"[K={K}] ")
        row = sm.loc[["Total relevant cost (£)", "Fill rate (units served / demand)", "Stock-out SKU-weeks"]]
        row.to_csv(REPORTS / f"sensitivity_K{K:g}.csv")
        print(row.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main([float(v) for v in sys.argv[1:]] or [5.0, 30.0])
