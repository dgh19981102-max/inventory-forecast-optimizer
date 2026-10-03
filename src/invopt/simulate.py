"""Rolling-horizon simulation of replenishment policies on the demand that actually happened.

Every simulated week t:
  1. the policy decides how much to order, using only information up to week t-1
     (on-hand stock, open orders, demand history, forecasts made at origin t-1);
  2. arrivals for week t are received, the order is placed (arrives in week t + L);
  3. the *real* historical demand of week t hits the shelf; unmet demand is lost;
  4. costs are booked: fixed ordering, holding on end-of-week stock, lost-sale penalty,
     overflow space above warehouse capacity.
Only the first week of each optimised plan is executed; next week we re-plan (rolling horizon).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import ErrorLibrary, quantile_scenarios
from .optimize import PlanProblem, safety_stock, solve_plan


# --------------------------------------------------------------------------- state & policies
@dataclass
class State:
    t: int                     # current week index (absolute)
    on_hand: np.ndarray        # (N,) before this week's arrivals
    arrivals: np.ndarray       # (T_total + Lmax + 1, N) scheduled arrivals by absolute week


class RuleOfThumb:
    """What many small retailers do: 'order enough to cover last month's sales'.

    Every `review` weeks: target stock position = average weekly demand of the last 4 weeks
    x (lead time + review period + buffer weeks). Order the gap, rounded up to full cases and
    at least the supplier MOQ. If the weekly budget is exceeded, all orders are scaled down.
    """
    name = "Rule of thumb"

    def __init__(self, review: int = 1, buffer_weeks: float = 1.0):
        self.review, self.buffer = review, buffer_weeks

    def decide(self, st: State, Y: np.ndarray, P: pd.DataFrame, settings: dict, t0: int) -> np.ndarray:
        N = Y.shape[1]
        if (st.t - t0) % self.review:
            return np.zeros(N)
        L = P["lead_time_weeks"].to_numpy(int)
        case, moq, cost = P["case_size"].to_numpy(), P["moq"].to_numpy(), P["unit_cost"].to_numpy()
        avg = Y[st.t - 4: st.t].mean(axis=0)
        target = avg * (L + self.review + self.buffer)
        position = st.on_hand + st.arrivals[st.t:].sum(axis=0)
        q = np.ceil(np.clip(target - position, 0, None) / case) * case
        q = np.where((q > 0) & (q < moq), moq, q)
        spend = (q * cost).sum()
        if spend > settings["weekly_budget"]:
            q = np.floor(q * settings["weekly_budget"] / spend / case) * case
            q = np.where(q < moq, 0, q)
        return q


class OptimisedPolicy:
    """Rolling-horizon MILP. `scenarios=False`: point forecast + safety stock.
    `scenarios=True`: S demand scenarios from the empirical forecast-error library."""

    def __init__(self, forecasts: pd.DataFrame, lib: ErrorLibrary, model: str, scenarios: bool,
                 n_scen: int = 20, seed: int = 42, verbose: bool = False,
                 scenario_mode: str = "quantile"):
        self.fc = forecasts[forecasts["model"] == model]
        self.fc_by_origin = {o: g.pivot(index="h", columns="sku", values="forecast") for o, g in self.fc.groupby("origin")}
        self.lib, self.scen, self.n_scen = lib, scenarios, n_scen
        self.rng = np.random.default_rng(seed)
        self.name = "Optimised: scenarios (uncertainty-aware)" if scenarios else "Optimised: point forecast + safety stock"
        self.solve_times: list[float] = []
        self.gaps: list[float] = []
        self.verbose = verbose
        self.scenario_mode = scenario_mode

    def decide(self, st: State, Y: np.ndarray, P: pd.DataFrame, settings: dict, t0: int) -> np.ndarray:
        o = st.t - 1
        H = settings["planning_horizon_weeks"]
        point = self.fc_by_origin[o][P.index].to_numpy()[:H]               # (H, N)
        Lmax = int(P["lead_time_weeks"].max())
        pipeline = st.arrivals[st.t: st.t + Lmax].T                         # (N, Lmax)
        if self.scen:
            if self.scenario_mode == "quantile":
                demand, probs = quantile_scenarios(self.lib, point, o, list(P.index))
            else:
                demand, probs = self.lib.scenarios(point, o, self.n_scen, self.rng, skus=list(P.index)), None
            ss = None
        else:
            demand, probs = point[None], None
            # per-SKU sigma of 1-week-ahead error, from past backtest residuals only
            past = self.fc[(self.fc["h"] == 1) & (self.fc["origin"] + 1 <= o) & ~self.fc["closed_week"]]
            sigma = (past["actual"] - past["forecast"]).groupby(past["sku"]).std().reindex(P.index).fillna(0).to_numpy()
            ss = safety_stock(P, sigma, settings["service_level_target"])
        pb = PlanProblem(params=P, on_hand=st.on_hand, pipeline=pipeline, demand=demand,
                         capacity=settings["warehouse_capacity_space"], budget=settings["weekly_budget"],
                         overflow_cost=settings["overflow_cost_per_space_week"], safety_stock=ss, probs=probs)
        res = solve_plan(pb)
        self.solve_times.append(res.solve_seconds)
        self.gaps.append(res.expected["mip_gap"])
        if self.verbose:
            print(f"  week {st.t}: solved in {res.solve_seconds:.1f}s, gap {res.expected['mip_gap']:.2%}", flush=True)
        return res.orders[0]


# --------------------------------------------------------------------------- simulator
def simulate(policy, Y: np.ndarray, P: pd.DataFrame, settings: dict, t0: int, t1: int,
             weeks: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate weeks t0..t1 (inclusive). Returns (per sku-week log, per week log)."""
    N = Y.shape[1]
    L = P["lead_time_weeks"].to_numpy(int)
    h, p = P["holding_cost_week"].to_numpy(), P["lost_sale_penalty"].to_numpy()
    K, cost, space = P["fixed_order_cost"].to_numpy(), P["unit_cost"].to_numpy(), P["space_per_unit"].to_numpy()
    cap, over_c = settings["warehouse_capacity_space"], settings["overflow_cost_per_space_week"]

    # identical starting point for every policy: (L + 1) weeks of recent average demand
    # on hand and one week of demand arriving in each week of the lead time
    avg = Y[t0 - 8: t0].mean(axis=0)
    arrivals = np.zeros((t1 + 10, N))
    for i in range(N):
        arrivals[t0: t0 + L[i], i] = np.round(avg[i])
    st = State(t=t0, on_hand=np.round(avg * (L + 1)), arrivals=arrivals)

    rows, wk = [], []
    for t in range(t0, t1 + 1):
        st.t = t
        q = policy.decide(st, Y, P, settings, t0)
        for i in range(N):
            if q[i] > 0:
                st.arrivals[t + L[i], i] += q[i]
        on_hand = st.on_hand + st.arrivals[t]
        space_used = float((on_hand * space).sum())
        overflow = max(0.0, space_used - cap)
        d = Y[t]
        sales = np.minimum(on_hand, d)
        lost = d - sales
        end = on_hand - sales
        st.on_hand = end
        rows.append(pd.DataFrame({
            "week": weeks[t], "sku": P.index, "demand": d, "order_qty": q, "received": st.arrivals[t],
            "sales": sales, "lost": lost, "end_inventory": end,
            "ordering_cost": K * (q > 0), "holding_cost": h * end, "lost_sale_cost": p * lost,
            "purchase_spend": cost * q, "inventory_value": cost * end, "cogs": cost * sales,
        }))
        wk.append({"week": weeks[t], "space_used": space_used, "overflow_space": overflow,
                   "overflow_cost": overflow * over_c})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(wk)


def summarise(log: pd.DataFrame, wk: pd.DataFrame) -> pd.Series:
    n_weeks = log["week"].nunique()
    inv_val = log.groupby("week")["inventory_value"].sum().mean()
    total = (log[["ordering_cost", "holding_cost", "lost_sale_cost"]].sum().sum() + wk["overflow_cost"].sum())
    return pd.Series({
        "Total relevant cost (£)": total,
        "  Ordering (£)": log["ordering_cost"].sum(),
        "  Holding (£)": log["holding_cost"].sum(),
        "  Lost-sale penalty (£)": log["lost_sale_cost"].sum(),
        "  Overflow space (£)": wk["overflow_cost"].sum(),
        "Lost units": log["lost"].sum(),
        "Fill rate (units served / demand)": log["sales"].sum() / log["demand"].sum(),
        "Stock-out SKU-weeks": int((log["lost"] > 0).sum()),
        "Orders placed": int((log["order_qty"] > 0).sum()),
        "Average inventory value (£)": inv_val,
        "Inventory turnover (annualised)": log["cogs"].sum() / inv_val * 52 / n_weeks,
        "Weeks over warehouse capacity": int((wk["overflow_space"] > 0).sum()),
        "Purchase spend (£)": log["purchase_spend"].sum(),
    })
