"""End-to-end planner used by the Excel CLI and the Streamlit app.

    workbook -> backtest forecasting models on the client's own history -> pick best
    -> forecast + error scenarios -> MILP order plan -> compare with rule of thumb.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .excel_io import PlanInput
from .forecast import ErrorLibrary, accuracy, quantile_scenarios, rolling_forecasts
from .optimize import PlanProblem, safety_stock, solve_plan


# --------------------------------------------------------------------------- rule of thumb as a plan
def rule_of_thumb_plan(P: pd.DataFrame, history: pd.DataFrame, on_hand, pipeline, H: int, budget: float,
                       buffer_weeks: float = 1.0) -> np.ndarray:
    """Order-up-to (lead time + 1 + buffer) weeks of last-4-week average demand, every week,
    projecting stock forward with that same average. Whole packs, MOQ, budget cap."""
    N = len(P)
    L = P["lead_time_weeks"].to_numpy(int)
    case, moq, cost = P["case_size"].to_numpy(), P["moq"].to_numpy(), P["unit_cost"].to_numpy()
    avg = history.iloc[-4:].mean().to_numpy()
    target = avg * (L + 1 + buffer_weeks)
    position = on_hand + pipeline.sum(axis=1)
    orders = np.zeros((H, N))
    for t in range(H):
        q = np.ceil(np.clip(target - position, 0, None) / case) * case
        q = np.where((q > 0) & (q < moq), moq, q)
        spend = (q * cost).sum()
        if spend > budget:
            q = np.floor(q * budget / spend / case) * case
            q = np.where(q < moq, 0, q)
        q = np.where(t + L < H, q, 0)          # same convention as the optimiser
        orders[t] = q
        position = position + q - avg
    return orders


# --------------------------------------------------------------------------- scoring a fixed plan
def evaluate_plan(orders: np.ndarray, P: pd.DataFrame, on_hand, pipeline, demand: np.ndarray,
                  capacity: float, overflow_cost: float) -> dict:
    """Play a fixed order plan against S demand scenarios (S, H, N), lost-sales dynamics."""
    S, H, N = demand.shape
    L = P["lead_time_weeks"].to_numpy(int)
    h, p = P["holding_cost_week"].to_numpy(), P["lost_sale_penalty"].to_numpy()
    K, cost, space = P["fixed_order_cost"].to_numpy(), P["unit_cost"].to_numpy(), P["space_per_unit"].to_numpy()
    arr = np.zeros((H, N))
    for i in range(N):
        for t in range(H):
            if t < L[i]:
                arr[t, i] = pipeline[i, t] if t < pipeline.shape[1] else 0
            else:
                arr[t, i] = orders[t - L[i], i]
    inv = np.repeat(np.asarray(on_hand, float)[None], S, axis=0)
    hold = np.zeros(S); lostc = np.zeros(S); over = np.zeros(S); lost_u = np.zeros(S); so = np.zeros(S)
    inv_val = np.zeros(S)
    for t in range(H):
        stock = inv + arr[t]
        over += np.clip((stock * space).sum(axis=1) - capacity, 0, None) * overflow_cost
        sales = np.minimum(stock, demand[:, t])
        lost = demand[:, t] - sales
        inv = stock - sales
        hold += (inv * h).sum(axis=1)
        lostc += (lost * p).sum(axis=1)
        lost_u += lost.sum(axis=1)
        so += (lost > 0.5).sum(axis=1)
        inv_val += (inv * cost).sum(axis=1) / H
    ordering = float((K * (orders > 0)).sum())
    total = ordering + hold + lostc + over
    return {
        "Expected total cost": total.mean(),
        "Bad-case total cost (P90)": np.quantile(total, 0.9),
        "  Ordering": ordering,
        "  Holding": hold.mean(),
        "  Lost-sale penalty": lostc.mean(),
        "  Overflow space": over.mean(),
        "Expected fill rate": 1 - lost_u.mean() / demand.sum(axis=(1, 2)).mean(),
        "Expected stock-out SKU-weeks": so.mean(),
        "Order lines": int((orders > 0).sum()),
        "Purchase spend": float((orders * cost).sum()),
        "Average inventory value": inv_val.mean(),
    }


# --------------------------------------------------------------------------- main entry
def make_plan(inp: PlanInput, settings_override: dict | None = None, models=("MovingAvg4", "ETS", "LightGBM", "Ensemble"),
              n_backtest_origins: int = 26, backtest: pd.DataFrame | None = None, progress=None) -> dict:
    st = {**inp.settings, **(settings_override or {})}
    P = inp.products.copy()
    hist = inp.history
    H = int(st["planning_horizon_weeks"])
    rng = np.random.default_rng(int(st["random_seed"]))
    Y = hist.to_numpy()
    T = len(Y)
    o_now = T - 1
    say = progress or (lambda msg: None)

    # 1) rolling backtest on the client's own history
    if backtest is None:
        say("Backtesting forecast models ...")
        n_bt = n_backtest_origins
        origins = range(max(13, o_now - H - n_bt + 1), o_now + 1)
        backtest = rolling_forecasts(Y, hist.index, origins, H=H, skus=list(P.index), models=list(models))
    acc = accuracy(backtest[backtest["origin"] + H <= o_now])
    model = acc["WAPE"].idxmin()

    # 2) forecast + uncertainty
    say(f"Forecasting with {model} ...")
    fc = backtest[(backtest["model"] == model) & (backtest["origin"] == o_now)]
    point = fc.pivot(index="h", columns="sku", values="forecast")[list(P.index)].to_numpy()[:H]
    lib = ErrorLibrary(backtest, model, H)
    q = lib.quantiles(o_now, (0.1, 0.9))
    opt_scen, probs = quantile_scenarios(lib, point, o_now, list(P.index))
    # evaluation scenarios: 500 random draws of each SKU's own past error paths (not the
    # quantile scenarios the optimiser saw)
    eval_scen = np.empty((500, H, len(P)))
    for j, sku in enumerate(P.index):
        paths = lib.available(o_now, sku)
        paths = paths if len(paths) >= 8 else lib.available(o_now)
        eval_scen[:, :, j] = point[None, :, j] * paths[rng.integers(0, len(paths), 500)][:, :H]

    on_hand = P["on_hand"].to_numpy(float)
    pipeline = P[["on_order_week1", "on_order_week2"]].to_numpy(float)
    common = dict(params=P, on_hand=on_hand, pipeline=pipeline, capacity=st["warehouse_capacity_space"],
                  budget=st["weekly_budget"], overflow_cost=st["overflow_cost_per_space_week"])

    # 3) optimise
    say("Optimising (scenario MILP) ...")
    res_s = solve_plan(PlanProblem(demand=opt_scen, probs=probs, **common))
    past = backtest[(backtest["model"] == model) & (backtest["h"] == 1) & (backtest["origin"] + 1 <= o_now) & ~backtest["closed_week"]]
    sigma = (past["actual"] - past["forecast"]).groupby(past["sku"]).std().reindex(P.index).fillna(0).to_numpy()
    res_d = solve_plan(PlanProblem(demand=point[None], safety_stock=safety_stock(P, sigma, st["service_level_target"]), **common))
    rot = rule_of_thumb_plan(P, hist, on_hand, pipeline, H, st["weekly_budget"])

    # 4) compare on the same evaluation scenarios
    ev = {name: evaluate_plan(o, P, on_hand, pipeline, eval_scen, st["warehouse_capacity_space"], st["overflow_cost_per_space_week"])
          for name, o in [("Rule of thumb", rot), ("Optimised: point forecast + safety stock", res_d.orders),
                          ("Optimised: scenarios (recommended)", res_s.orders)]}
    comparison = pd.DataFrame(ev).round(2)
    base = comparison.loc["Expected total cost", "Rule of thumb"]
    comparison.loc["Saving vs rule of thumb"] = [
        f"{(base - v) / base:.1%}" if base > 0 else "" for v in comparison.loc["Expected total cost"]]
    comparison.index.name = "metric"

    weeks = pd.date_range(hist.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="7D")
    L = P["lead_time_weeks"].to_numpy(int)
    detail = []
    for i, sku in enumerate(P.index):
        for t in range(H):
            arriving = pipeline[i, t] if t < L[i] and t < pipeline.shape[1] else (res_s.orders[t - L[i], i] if t >= L[i] else 0)
            detail.append({
                "week_start": weeks[t], "sku": sku, "description": P.loc[sku, "description"],
                "forecast_P10": round(point[t, i] * q[0, t], 1), "forecast_P50": round(point[t, i], 1),
                "forecast_P90": round(point[t, i] * q[1, t], 1), "order_qty": res_s.orders[t, i],
                "arriving_qty": arriving,
                "projected_stock_end_of_week": round(float(res_s.expected["projected_inventory"][t, i]), 1),
            })
    return {
        "settings": st, "products": P, "weeks": weeks, "model": model, "model_wape": float(acc.loc[model, "WAPE"]),
        "accuracy": acc, "orders": res_s.orders, "orders_point": res_d.orders, "orders_rule": rot,
        "solve_seconds": res_s.solve_seconds, "comparison": comparison, "detail": pd.DataFrame(detail),
        "point": point, "interval_ratio": q, "backtest": backtest, "n_eval_scenarios": eval_scen.shape[0],
    }
