"""Replenishment MILP (OR-Tools ModelBuilder + open-source SCIP / HiGHS).

One model covers both versions:
* deterministic  -> 1 demand scenario (the point forecast) + soft safety-stock target
* scenario-based -> S demand scenarios sampled from the forecast-error library; one
  order plan must work for all of them (orders are "here-and-now" decisions,
  inventory and lost sales are computed per scenario) and the *expected* cost is minimised.

Timeline of week t (t = 0 is the current week)
    start of week: receive arrivals A[i,t]  ->  place order Q[i,t] (arrives in week t + L_i)
    during week:   demand D[s,i,t]; what cannot be served is lost (U)
    end of week:   pay holding cost on inventory I[s,i,t]

Decision variables
    n[i,t]  integer   number of cases ordered            Q[i,t] = case_i * n[i,t]
    y[i,t]  binary    1 if SKU i is ordered in week t     (fixed cost + minimum order quantity)
    I[s,i,t] >= 0     end-of-week inventory in scenario s
    U[s,i,t] >= 0     lost sales in scenario s
    O[s,t]   >= 0     storage space used above warehouse capacity (rented overflow, soft limit)
    V[i,t]   >= 0     shortfall below safety stock (deterministic version only)

Constraints
    inventory balance  I[s,i,t] = I[s,i,t-1] + A[i,t] - D[s,i,t] + U[s,i,t]
    lost sales         U[s,i,t] <= D[s,i,t]
    warehouse          sum_i space_i (I[s,i,t-1] + A[i,t]) <= capacity + O[s,t]
    budget             sum_i cost_i Q[i,t] <= weekly budget
    MOQ / fixed cost   moq_i y[i,t] <= Q[i,t] <= M[i,t] y[i,t]
    lead time          A[i,t] = Q[i,t-L_i] (or already-open orders for t < L_i)
    safety stock       I[0,i,t] + V[i,t] >= SS_i     (deterministic version)

Objective (minimise expected relevant cost)
    sum K_i y[i,t] + 1/S sum_s [ sum h_i I + p_i U + c_over O ] + sum ss_pen_i V
Unit purchase cost is not in the objective: every unit sold is bought exactly once at the
same price whatever the timing, so it does not change the decision. Cash is controlled by the
weekly budget constraint and purchase spend is reported in the results.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.linear_solver.python import model_builder as mb


@dataclass
class PlanProblem:
    params: pd.DataFrame          # per-SKU table (see params.build_sku_params)
    on_hand: np.ndarray           # (N,) units on hand at start of week 0 (before arrivals)
    pipeline: np.ndarray          # (N, k) open orders: pipeline[i, k] arrives at start of week k
    demand: np.ndarray            # (S, H, N) demand scenarios (S = 1 for deterministic)
    capacity: float               # storage space units
    budget: float                 # purchase budget per week (on order placement)
    overflow_cost: float          # per space unit above capacity per week
    probs: np.ndarray | None = None          # (S,) scenario probabilities (default: equally likely)
    safety_stock: np.ndarray | None = None   # (N,) only for the deterministic version
    ss_penalty_share: float = 0.5            # SS shortfall penalty = share x lost-sale penalty
    time_limit: float = 30.0
    # Relative gap must be small: the objective contains a large, unavoidable lost-sales part from
    # the extreme scenarios, so a 1% gap would let the solver ignore single-SKU decisions worth £20-50.
    mip_gap: float = 0.001
    solver: str = "SCIP"                     # or "HIGHS" (both open source, bundled with OR-Tools)
    # --- speed-ups for the rolling-horizon setting (only week 0 of a plan is executed) ---
    binary_weeks: int = 99                   # order/no-order exact for weeks < binary_weeks (lower = faster, approx.)
    integer_weeks: int = 1                   # whole cases enforced for weeks < integer_weeks
    capacity_per_scenario: bool = False      # False: capacity checked on scenario-average stock
    # Recourse (two-stage): only THIS week's order is shared by all scenarios and later orders may
    # differ per scenario. Tested in the simulation: it solves ~5x faster but over-stocks (it counts
    # on perfect later reactions, rented overflow space ballooned), so it is OFF by default.
    recourse: bool = False


@dataclass
class PlanResult:
    orders: np.ndarray            # (H, N) units ordered per week
    objective: float
    solve_seconds: float
    status: str
    expected: dict = field(default_factory=dict)


def solve_plan(pb: PlanProblem) -> PlanResult:
    P = pb.params
    S, H, N = pb.demand.shape
    D = pb.demand
    L = P["lead_time_weeks"].to_numpy(int)
    case = P["case_size"].to_numpy(int)
    moq = np.maximum(P["moq"].to_numpy(float), case)
    cost = P["unit_cost"].to_numpy(float)
    space = P["space_per_unit"].to_numpy(float)
    h = P["holding_cost_week"].to_numpy(float)
    p = P["lost_sale_penalty"].to_numpy(float)
    K = P["fixed_order_cost"].to_numpy(float)
    ss = pb.safety_stock if pb.safety_stock is not None else np.zeros(N)
    w = np.full(S, 1.0 / S) if pb.probs is None else np.asarray(pb.probs, float) / np.sum(pb.probs)

    # An order placed in week t only matters if it arrives inside the horizon (t + L < H).
    can_order = [(i, t) for i in range(N) for t in range(H) if t + L[i] < H]
    # Tight big-M: an order never needs to exceed the largest scenario demand from its arrival
    # week to the end of the horizon (+ safety stock), nor what the budget can pay for.
    tail = np.flip(np.cumsum(np.flip(D, axis=1), axis=1), axis=1).max(axis=0)   # (H, N)
    nmax = {(i, t): max(1, math.ceil(max(min(tail[t + L[i], i] + ss[i], pb.budget / cost[i]), moq[i]) / case[i]))
            for (i, t) in can_order}

    m = mb.Model()
    rec = pb.recourse and S > 1
    # Order variables are indexed (s, i, t). Week 0 (and every week if no recourse) uses one shared
    # variable for all scenarios (key s = -1): the decision we actually execute must not depend on
    # demand we have not seen yet ("non-anticipativity").
    def key(s, i, t):
        return (s, i, t) if (rec and t >= 1) else (-1, i, t)
    keys = sorted({key(s, i, t) for s in range(S) for (i, t) in can_order})
    # Relax-and-fix: the week-0 decision is exact (whole cases, true order/no-order binary). Later
    # weeks are re-optimised next week anyway; in recourse mode they are continuous (LP), which
    # keeps the model at ~30 binaries and solves in seconds.
    def exact_int(k):
        return k[2] < pb.integer_weeks and k[0] == -1
    def exact_bin(k):
        return k[2] < pb.binary_weeks and k[0] == -1
    n = {k: (m.new_int_var if exact_int(k) else m.new_num_var)(0, nmax[k[1], k[2]], f"n_{k}") for k in keys}
    y = {k: (m.new_bool_var(f"y_{k}") if exact_bin(k) else m.new_num_var(0, 1, f"y_{k}")) for k in keys}
    prob = {k: (w[k[0]] if k[0] >= 0 else 1.0) for k in keys}
    inv = {(s, i, t): m.new_num_var(0, math.inf, f"I_{s}_{i}_{t}") for s in range(S) for i in range(N) for t in range(H)}
    lost = {(s, i, t): m.new_num_var(0, float(D[s, t, i]), f"U_{s}_{i}_{t}") for s in range(S) for i in range(N) for t in range(H)}
    capS = list(range(S)) if pb.capacity_per_scenario else [0]
    over = {(s, t): m.new_num_var(0, math.inf, f"O_{s}_{t}") for s in capS for t in range(H)}

    def arrivals(i, t, s=0):
        if t < L[i]:
            return float(pb.pipeline[i, t]) if t < pb.pipeline.shape[1] else 0.0
        k = key(s, i, t - L[i])
        return case[i] * n[k] if k in n else 0.0

    def inv_prev(s, i, t):
        return float(pb.on_hand[i]) if t == 0 else inv[s, i, t - 1]

    for s in range(S):
        for t in range(H):
            for i in range(N):
                m.add(inv[s, i, t] == inv_prev(s, i, t) + arrivals(i, t, s) - float(D[s, t, i]) + lost[s, i, t])
            if pb.capacity_per_scenario:
                m.add(mb.LinearExpr.sum([space[i] * (inv_prev(s, i, t) + arrivals(i, t, s)) for i in range(N)])
                      <= pb.capacity + over[s, t])
    if not pb.capacity_per_scenario:
        # warehouse check on the scenario-average stock (one constraint per week); the simulator
        # still charges overflow on the stock that really happens
        for t in range(H):
            m.add(mb.LinearExpr.sum([(space[i] * w[s]) * (inv_prev(s, i, t) + arrivals(i, t, s)) for s in range(S) for i in range(N)])
                  <= pb.capacity + over[0, t])
    for sv in sorted({k[0] for k in keys}):
        for t in range(H):
            items = [cost[i] * case[i] * n[k] for k in keys if k[0] == sv and k[2] == t for i in [k[1]]]
            if not items:
                continue
            m.add(mb.LinearExpr.sum(items) <= pb.budget)
    for k in keys:
        i, t = k[1], k[2]
        m.add(case[i] * n[k] >= moq[i] * y[k])
        m.add(n[k] <= nmax[i, t] * y[k])

    obj = [prob[k] * K[k[1]] * y[k] for k in keys]
    obj += [(h[i] * w[s]) * inv[s, i, t] + (p[i] * w[s]) * lost[s, i, t] for s in range(S) for i in range(N) for t in range(H)]
    obj += [pb.overflow_cost * (w[s] if pb.capacity_per_scenario else 1.0) * over[s, t] for s in capS for t in range(H)]
    ssv = {}
    if pb.safety_stock is not None:
        for i in range(N):
            for t in range(L[i], H):           # weeks an order placed now can still influence
                ssv[i, t] = m.new_num_var(0, math.inf, f"V_{i}_{t}")
                m.add(inv[0, i, t] + ssv[i, t] >= float(ss[i]))
                obj.append(pb.ss_penalty_share * p[i] * ssv[i, t])
    m.minimize(mb.LinearExpr.sum(obj))

    solver = mb.Solver(pb.solver)
    solver.set_time_limit_in_seconds(pb.time_limit)
    if pb.solver == "SCIP":
        solver.set_solver_specific_parameters(f"limits/gap = {pb.mip_gap}")
    elif pb.solver == "HIGHS":
        solver.set_solver_specific_parameters(f"mip_rel_gap={pb.mip_gap}")
    t0 = time.perf_counter()
    status = solver.solve(m)
    dt = time.perf_counter() - t0
    if status not in (mb.SolveStatus.OPTIMAL, mb.SolveStatus.FEASIBLE):
        raise RuntimeError(f"No feasible plan found ({status})")

    # week 0 = the decision to execute; later weeks = probability-weighted indicative plan
    orders = np.zeros((H, N))
    for k, v in n.items():
        orders[k[2], k[1]] += prob[k] * case[k[1]] * solver.value(v)
    orders[1:] = np.round(orders[1:] / case) * case
    orders[0] = np.round(orders[0])
    I_ = np.array([[[solver.value(inv[s, i, t]) for i in range(N)] for t in range(H)] for s in range(S)])
    U_ = np.array([[[solver.value(lost[s, i, t]) for i in range(N)] for t in range(H)] for s in range(S)])
    O_ = np.array([[solver.value(over[s, t]) for t in range(H)] for s in capS])
    expected = {
        "ordering": float(sum(prob[k] * K[k[1]] * solver.value(y[k]) for k in keys)),
        "holding": float(np.einsum("s,sti,i->", w, I_, h)),
        "lost_sales_penalty": float(np.einsum("s,sti,i->", w, U_, p)),
        "overflow": float((O_ * (w[:, None] if pb.capacity_per_scenario else 1.0)).sum() * pb.overflow_cost),
        "lost_units": float(np.einsum("s,sti->", w, U_)),
        "projected_inventory": np.einsum("s,sti->ti", w, I_),       # (H, N) probability-weighted
        "mip_gap": float(abs(solver.objective_value - solver.best_objective_bound) / max(1e-9, abs(solver.objective_value))),
    }
    return PlanResult(orders=orders, objective=float(solver.objective_value), solve_seconds=dt,
                      status=str(status).split(".")[-1], expected=expected)


def safety_stock(params: pd.DataFrame, sigma_week: np.ndarray, service_level: float) -> np.ndarray:
    """Textbook safety stock: z(service level) x sigma(1-week forecast error) x sqrt(lead time + 1 review week)."""
    from scipy.stats import norm
    z = norm.ppf(service_level)
    L = params["lead_time_weeks"].to_numpy(float)
    return z * sigma_week * np.sqrt(L + 1)
