"""Are the demand scenarios honest? Calibration of scenario variants over Jun-Nov 2011.

For every SKU and week we build the scenarios a planner would have had, add up the next k weeks
(k = lead time + review week = 2..3) and check where the real demand fell:
  * above_P90   should be ~10%   (more = scenarios too optimistic -> stock-outs)
  * above_P97_5 should be ~2.5%
  * scen_mean_over_actual should be ~1.0 (below 1 = scenarios under-estimate demand on average)
"""
import numpy as np
import pandas as pd
from _common import H, PROCESSED, REPORTS, SIM_END, SIM_START, load_processed

from invopt.forecast import ErrorLibrary, accuracy, quantile_scenarios

VARIANTS = {
    "own SKU errors, raw": dict(errors="sku", center=None),
    "own SKU errors, mean-centred": dict(errors="sku", center="mean"),
    "own SKU errors, median-centred": dict(errors="sku", center="median"),
    "pooled errors, raw": dict(errors="pooled", center=None),
    "pooled errors, median-centred (used)": dict(errors="pooled", center="median"),
}


def main():
    weekly, _ = load_processed()
    Y, skus = weekly.to_numpy(), list(weekly.columns)
    bt = pd.read_parquet(PROCESSED / "backtest.parquet")
    model = accuracy(bt[bt["origin"] + H <= SIM_START - 1])["WAPE"].idxmin()
    lib = ErrorLibrary(bt, model, H)
    fc = bt[bt["model"] == model]
    pts = {o: g.pivot(index="h", columns="sku", values="forecast")[skus].to_numpy() for o, g in fc.groupby("origin")}
    rows = []
    for name, kw in VARIANTS.items():
        for k in (2, 3):
            a90 = a975 = n = 0
            num = den = 0.0
            for o in range(SIM_START - 1, SIM_END - k + 1):
                D, pr = quantile_scenarios(lib, pts[o], o, skus, **kw)
                cum = D[:, :k].sum(axis=1)                         # (K, N)
                act = Y[o + 1: o + 1 + k].sum(axis=0)              # (N,)
                for j in range(len(skus)):
                    order = np.argsort(cum[:, j]); c = cum[order, j]; cdf = np.cumsum(pr[order])
                    a90 += act[j] > c[min(len(c) - 1, np.searchsorted(cdf, 0.9))]
                    a975 += act[j] > c[min(len(c) - 1, np.searchsorted(cdf, 0.975))]
                    n += 1
                num += (cum * pr[:, None]).sum(); den += act.sum()
            rows.append({"variant": name, "weeks_summed": k, "above_P90 (target 0.10)": a90 / n,
                         "above_P97.5 (target 0.025)": a975 / n, "scenario mean / actual": num / den})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(REPORTS / "scenario_calibration.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
