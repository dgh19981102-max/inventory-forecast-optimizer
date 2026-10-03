"""Step 1 - raw UCI Online Retail II -> weekly demand of the 30 most stable SKUs.

Put the raw data in data/raw/ first (see README):
  * data/raw/online_retail_II.xlsx   (official UCI file, two sheets), or
  * data/raw/online_retail_II.csv    (CSV export of both sheets with a `source_sheet` column)
"""
import sys
import pandas as pd
from _common import PROCESSED, ROOT

from invopt.data import build_dataset, stats_table


def ensure_csv():
    raw = ROOT / "data" / "raw"
    csv, xlsx = raw / "online_retail_II.csv", raw / "online_retail_II.xlsx"
    if csv.exists():
        return csv
    if xlsx.exists():
        print("Converting the Excel file to CSV (one-off, ~2 min) ...")
        sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        df = pd.concat([d.assign(source_sheet=name) for name, d in sheets.items()], ignore_index=True)
        df.to_csv(csv, index=False)
        return csv
    sys.exit("Raw data not found. Download it from https://archive.ics.uci.edu/dataset/502/online+retail+ii "
             "and unzip online_retail_II.xlsx into data/raw/")


def main():
    path = ensure_csv()
    weekly, master = build_dataset(n_skus=30, raw_path=path)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "weekly_demand.csv")
    master.to_csv(PROCESSED / "sku_master.csv")
    print(master.join(stats_table(weekly)).to_string())
    print(f"\n{weekly.shape[0]} weeks ({weekly.index[0].date()} .. {weekly.index[-1].date()}), {weekly.shape[1]} SKUs")


if __name__ == "__main__":
    main()
