"""Static figures for the README (matplotlib, light theme)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#9a9890"
POLICY_COLORS = {"Rule of thumb": NEUTRAL,
                 "Optimised: point forecast + safety stock": SERIES[0],
                 "Optimised: scenarios (uncertainty-aware)": SERIES[1]}
MODEL_COLORS = {"SeasonalNaive": NEUTRAL, "MovingAvg4": SERIES[3], "ETS": SERIES[0],
                "LightGBM": SERIES[2], "Ensemble": SERIES[1]}


def _style(ax, title=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)


def _fig(w=9, h=4.6, **kw):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, ax


def forecast_accuracy_bars(acc: pd.DataFrame, path, subtitle: str):
    """acc: index model, column WAPE (overall)."""
    acc = acc.sort_values("WAPE", ascending=False)
    fig, ax = _fig(9, 3.8)
    y = np.arange(len(acc))
    best = acc["WAPE"].idxmin()
    cols = [SERIES[1] if m == best else (NEUTRAL if m in ("SeasonalNaive", "MovingAvg4") else SERIES[0]) for m in acc.index]
    ax.barh(y, acc["WAPE"] * 100, color=cols, height=0.6, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, acc["WAPE"]):
        ax.text(v * 100 + 0.8, yi, f"{v:.1%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, acc.index, color=INK)
    ax.set_xlim(0, acc["WAPE"].max() * 118)
    ax.set_xlabel("WAPE (%), 1-6 weeks ahead - lower is better", color=INK2)
    ax.grid(axis="x", color=GRID)
    _style(ax, "Forecast accuracy, rolling-origin backtest", subtitle)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def forecast_example(weeks, actual, fc, lo, hi, title, path):
    fig, ax = _fig()
    ax.fill_between(weeks, lo, hi, color=SERIES[0], alpha=0.15, lw=0, label="80% interval (P10-P90)")
    ax.plot(weeks, actual, color=INK, lw=1.6, label="Actual weekly demand")
    ax.plot(weeks, fc, color=SERIES[0], lw=2, label="1-week-ahead forecast")
    ax.set_ylabel("Units / week", color=INK2)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    _style(ax, title, "Forecast made each week using only data available at that time")
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def policy_cost_breakdown(summary: pd.DataFrame, path):
    """summary: columns = policies, rows include cost components & fill rate."""
    comps = ["  Ordering (£)", "  Holding (£)", "  Lost-sale penalty (£)", "  Overflow space (£)"]
    names = ["Ordering", "Holding", "Lost-sale penalty", "Overflow space"]
    cols = [SERIES[0], SERIES[2], SERIES[1], SERIES[3]]
    pols = list(summary.columns)
    fig, ax = _fig(9, 4.8)
    y = np.arange(len(pols))[::-1]
    left = np.zeros(len(pols))
    for c, n, col in zip(comps, names, cols):
        v = summary.loc[c].to_numpy(float)
        ax.barh(y, v, left=left, color=col, height=0.55, label=n, edgecolor=SURFACE, linewidth=2)
        left += v
    base = summary.loc["Total relevant cost (£)"].iloc[0]
    for yi, tot, fr in zip(y, summary.loc["Total relevant cost (£)"], summary.loc["Fill rate (units served / demand)"]):
        saving = "" if tot == base else f"  ({(base - tot) / base:+.0%} vs rule)".replace("+", "-")
        ax.text(tot + base * 0.01, yi, f"£{tot:,.0f}{saving}\nfill rate {fr:.1%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [p.replace("Optimised: ", "Optimised:\n") for p in pols], color=INK)
    ax.set_xlim(0, base * 1.35)
    ax.set_xlabel("Total relevant cost over 27 weeks (£)", color=INK2)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.32), labelcolor=INK2)
    _style(ax, "Replaying Jun-Nov 2011 real demand: cost by replenishment policy",
           "30 SKUs, weekly re-planning, same starting stock; costs under the stated assumptions")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def inventory_trajectory(logs: dict, sku: str, desc: str, path):
    fig, ax = _fig(9, 5.2)
    first = next(iter(logs.values()))
    d = first[first["sku"] == sku]
    ax.bar(d["week"], d["demand"], width=5, color=GRID, label="Actual demand")
    for name, log in logs.items():
        g = log[log["sku"] == sku]
        ax.plot(g["week"], g["end_inventory"], lw=2, color=POLICY_COLORS.get(name, INK2), label=f"Stock: {name}")
        so = g[g["lost"] > 0]
        ax.scatter(so["week"], np.zeros(len(so)), marker="v", s=40, color=POLICY_COLORS.get(name, INK2),
                   edgecolor=SURFACE, linewidth=1.5, zorder=5, clip_on=False)
    ax.set_ylabel("Units", color=INK2)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, labelcolor=INK2)
    _style(ax, f"End-of-week stock, {sku} {desc.title()}", "Triangles on the axis = weeks with a stock-out")
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)
