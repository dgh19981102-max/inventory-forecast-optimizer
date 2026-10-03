"""Cost & constraint assumptions for the fictional retailer.

The UCI data only contains *selling* prices and quantities. Everything below is an
ASSUMPTION, chosen to be plausible for a small UK gift/homeware wholesaler. All of
it is written to the Excel template, so a client simply overwrites it with real numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    cost_ratio: float = 0.55            # unit purchase cost = 55% of median selling price (45% gross margin)
    holding_rate_year: float = 0.26     # cost of capital + insurance + obsolescence, per year, on unit cost
    storage_cost_per_space_week: float = 0.004   # £ per storage-space unit per week
    goodwill_rate: float = 0.10         # extra lost-sale penalty as share of price (customer goes elsewhere)
    fixed_order_cost: float = 15.0      # £ per order line (admin, inbound handling, delivery share)
    moq_weeks: float = 1.0              # supplier minimum order ~ 1 week of average demand (rounded to case)
    capacity_weeks: float = 3.0         # warehouse holds ~3 weeks of average total volume
    budget_weeks: float = 1.6           # weekly purchasing budget = 1.6 x average weekly purchase cost
    overflow_cost_per_space_week: float = 0.05   # renting overflow space if capacity exceeded (soft limit)


# Storage space per unit, by product family (1.0 = one T-light holder). Assumed.
SPACE_BY_KEYWORD = [
    ("CABINET", 6.0),
    ("BUILDING BLOCK", 2.0),
    ("LUNCH BOX", 1.0),
    ("T-LIGHT", 1.0),
    ("TRINKET BOX", 0.6),
    ("CUP", 0.5),
    ("JUMBO", 0.4),            # folded bags
    ("STORAGE BAG", 0.4),
    ("LUNCH BAG", 0.4),
    ("CHARLOTTE BAG", 0.2),
    ("HANGING DECORATION", 0.3),
    ("CAKE CASES", 0.15),
    ("WRAP", 0.1),
    ("CARD", 0.05),
]

# Items assumed to be imported (2-week lead time); everything else 1 week.
IMPORTED_KEYWORDS = ("CABINET", "BUILDING BLOCK", "T-LIGHT")


def space_per_unit(description: str) -> float:
    d = description.upper()
    for kw, v in SPACE_BY_KEYWORD:
        if kw in d:
            return v
    return 1.0


def build_sku_params(master: pd.DataFrame, weekly: pd.DataFrame, a: Assumptions | None = None,
                     reference_weeks: slice | None = None) -> pd.DataFrame:
    """Per-SKU parameter table (this is exactly the 'Products' sheet of the Excel template).

    `reference_weeks` = history used to size MOQ (should be data available before the plan starts).
    """
    a = a or Assumptions()
    hist = weekly if reference_weeks is None else weekly.iloc[reference_weeks]
    hist = hist[hist.sum(axis=1) > 0]
    mean_demand = hist.mean()
    p = pd.DataFrame(index=master.index)
    p["description"] = master["description"].str.strip()
    p["sell_price"] = master["sell_price"].round(2)
    p["unit_cost"] = (a.cost_ratio * p["sell_price"]).round(3)
    p["space_per_unit"] = p["description"].map(space_per_unit)
    p["holding_cost_week"] = (p["unit_cost"] * a.holding_rate_year / 52 + a.storage_cost_per_space_week * p["space_per_unit"]).round(4)
    p["lost_sale_penalty"] = (p["sell_price"] - p["unit_cost"] + a.goodwill_rate * p["sell_price"]).round(3)
    p["case_size"] = master["case_size"].astype(int)
    moq = np.ceil(a.moq_weeks * mean_demand / p["case_size"]) * p["case_size"]
    p["moq"] = moq.astype(int)
    p["lead_time_weeks"] = [2 if any(k in d.upper() for k in IMPORTED_KEYWORDS) else 1 for d in p["description"]]
    p["fixed_order_cost"] = a.fixed_order_cost
    p.index.name = "sku"
    return p


def global_settings(sku_params: pd.DataFrame, weekly: pd.DataFrame, a: Assumptions | None = None,
                    reference_weeks: slice | None = None) -> dict:
    a = a or Assumptions()
    hist = weekly if reference_weeks is None else weekly.iloc[reference_weeks]
    hist = hist[hist.sum(axis=1) > 0]
    mean = hist.mean()[sku_params.index]
    weekly_space = float((mean * sku_params["space_per_unit"]).sum())
    weekly_cost = float((mean * sku_params["unit_cost"]).sum())
    return {
        "warehouse_capacity_space": round(a.capacity_weeks * weekly_space, -1),
        "weekly_budget": round(a.budget_weeks * weekly_cost, -1),
        "overflow_cost_per_space_week": a.overflow_cost_per_space_week,
        "planning_horizon_weeks": 6,
        "service_level_target": 0.95,
        "random_seed": 42,
    }


def assumptions_table(a: Assumptions | None = None) -> pd.DataFrame:
    a = a or Assumptions()
    return pd.Series(asdict(a), name="value").rename_axis("assumption").to_frame()
