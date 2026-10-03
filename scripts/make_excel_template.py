"""Build examples/input_template.xlsx from the processed UCI data (full history up to 2011-11-28)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invopt.excel_io import write_template  # noqa: E402
from invopt.params import build_sku_params, global_settings  # noqa: E402


def main(out=ROOT / "examples" / "input_template.xlsx"):
    weekly = pd.read_csv(ROOT / "data/processed/weekly_demand.csv", index_col=0, parse_dates=True)
    master = pd.read_csv(ROOT / "data/processed/sku_master.csv", index_col=0, dtype={0: str})
    weekly.columns = weekly.columns.astype(str); master.index = master.index.astype(str)
    P = build_sku_params(master, weekly)
    settings = global_settings(P, weekly)
    # Current stock position (assumed): ~ (lead time + 0.5) weeks of the last 8 weeks' average demand,
    # plus one week of demand already on order for each lead-time week.
    recent = weekly.iloc[-8:].mean()
    P["on_hand"] = np.round(recent * (P["lead_time_weeks"] + 0.5)).astype(int)
    P["on_order_week1"] = np.round(recent).astype(int)
    P["on_order_week2"] = np.where(P["lead_time_weeks"] >= 2, np.round(recent), 0).astype(int)
    write_template(out, settings, P, weekly)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
