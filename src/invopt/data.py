"""Data preparation: raw UCI Online Retail II transactions -> clean weekly demand per SKU.

Pipeline
--------
1. Load the raw transaction file (CSV export of the two Excel sheets).
2. Remove the 9-day overlap between the two sheets (Dec 1-9 2010 appears in both).
3. Drop cancellations (invoice starting with "C"), returns (qty <= 0), zero prices,
   exact duplicate lines and non-product codes (postage, fees, manual adjustments...).
4. Aggregate to weekly quantity per SKU (weeks start on Monday). Weeks with no
   sales get 0 (e.g. the shop's Christmas closure).
5. Select SKUs with long, stable sales history.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "online_retail_II.csv"

# The last week in the data (2011-12-05) is partial (data ends Friday 9 Dec noon) -> excluded.
FIRST_WEEK = pd.Timestamp("2009-11-30")
LAST_WEEK = pd.Timestamp("2011-11-28")

PRODUCT_CODE_REGEX = r"^\d{5}[A-Za-z]{0,2}$"  # e.g. 85123A; excludes POST, DOT, M, BANK CHARGES ...


def load_transactions(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Load and clean raw transactions. Returns one row per (cleaned) invoice line."""
    df = pd.read_csv(
        path,
        dtype={"Invoice": str, "StockCode": str, "Description": str},
        parse_dates=["InvoiceDate"],
    )
    if "source_sheet" in df.columns:
        # Sheet "Year 2009-2010" runs until 2010-12-09 and overlaps with sheet "Year 2010-2011".
        first = df["source_sheet"].str.contains("2009")
        df = df[~first | (df["InvoiceDate"] < "2010-12-01")]
        df = df.drop(columns="source_sheet")
    df = df.drop_duplicates()
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["StockCode"].str.match(PRODUCT_CODE_REGEX)]
    df["StockCode"] = df["StockCode"].str.upper()
    return df.reset_index(drop=True)


def weekly_demand(tx: pd.DataFrame) -> pd.DataFrame:
    """Wide table: index = week start (Monday), columns = StockCode, values = units sold."""
    week = tx["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    w = tx.assign(week=week).groupby(["week", "StockCode"])["Quantity"].sum().unstack(fill_value=0)
    full_index = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="W-MON")
    return w.reindex(full_index, fill_value=0).rename_axis("week").astype(float)


def sku_attributes(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU description, median selling price and typical pack size (most common line qty)."""
    g = tx.groupby("StockCode")
    out = pd.DataFrame(
        {
            "description": g["Description"].agg(lambda s: s.mode().iat[0] if s.notna().any() else ""),
            "sell_price": g["Price"].median(),
            "case_size": g["Quantity"].agg(lambda s: int(s.mode().iat[0])),
        }
    )
    return out


def select_stable_skus(
    weekly: pd.DataFrame,
    n: int = 30,
    min_active_share: float = 0.95,
    min_mean: float = 40.0,
) -> list[str]:
    """Pick SKUs with long, stable history.

    Criteria (all computed on weekly units):
      * sold in >= 95% of weeks (ignoring the all-shop closure weeks) -> full life cycle covered;
      * mean >= 40 units/week -> enough volume to matter;
      * rank by coefficient of variation (std / mean) and keep the n most stable.
    """
    open_weeks = weekly.sum(axis=1) > 0
    w = weekly[open_weeks]
    active = (w > 0).mean()
    mean = w.mean()
    cv = w.std() / mean
    ok = (active >= min_active_share) & (mean >= min_mean)
    return cv[ok].sort_values().index[:n].tolist()


def build_dataset(n_skus: int = 30, raw_path: str | Path = RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (weekly demand for selected SKUs, SKU master data)."""
    tx = load_transactions(raw_path)
    weekly = weekly_demand(tx)
    skus = select_stable_skus(weekly, n=n_skus)
    attrs = sku_attributes(tx[tx["StockCode"].isin(skus)]).loc[skus]
    return weekly[skus], attrs


def stats_table(weekly: pd.DataFrame) -> pd.DataFrame:
    open_weeks = weekly.sum(axis=1) > 0
    w = weekly[open_weeks]
    return pd.DataFrame({"mean": w.mean(), "cv": w.std() / w.mean(), "active_share": (w > 0).mean()}).round(3)


if __name__ == "__main__":  # quick look
    d, a = build_dataset()
    print(a.join(stats_table(d)))
    print(np.round(d.sum(axis=1).describe(), 1))
