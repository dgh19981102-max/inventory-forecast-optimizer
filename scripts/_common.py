"""Shared paths and loaders for the scripts."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

H = 6                # planning / forecast horizon (weeks)
FIRST_ORIGIN = 52    # first backtest origin (needs a year of history for seasonal features)
SIM_START = 78       # first simulated week index = 2011-05-30 ... last = 104 (2011-11-28)
SIM_END = 104


def load_processed():
    weekly = pd.read_csv(PROCESSED / "weekly_demand.csv", index_col=0, parse_dates=True)
    master = pd.read_csv(PROCESSED / "sku_master.csv", index_col=0, dtype={0: str})
    weekly.columns = weekly.columns.astype(str)
    master.index = master.index.astype(str)
    return weekly, master
