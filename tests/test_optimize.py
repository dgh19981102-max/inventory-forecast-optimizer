import numpy as np
import pandas as pd
import pytest

from invopt.optimize import PlanProblem, solve_plan
from invopt.planner import evaluate_plan


def params(n=2, L=1, K=10.0):
    return pd.DataFrame({
        "description": [f"p{i}" for i in range(n)], "sell_price": 2.0, "unit_cost": 1.0,
        "space_per_unit": 1.0, "holding_cost_week": 0.01, "lost_sale_penalty": 1.2,
        "case_size": 10, "moq": 20, "lead_time_weeks": L, "fixed_order_cost": K,
    }, index=[f"s{i}" for i in range(n)])


def problem(P, demand, **kw):
    N = len(P)
    base = dict(params=P, on_hand=np.zeros(N), pipeline=np.zeros((N, 2)), demand=demand,
                capacity=1e6, budget=1e6, overflow_cost=1.0)
    base.update(kw)
    return PlanProblem(**base)


def test_orders_respect_case_moq_budget():
    P = params()
    D = np.full((1, 4, 2), 13.0)
    res = solve_plan(problem(P, D, budget=40.0))
    q = res.orders
    assert np.all(q % 10 == 0)                              # whole cases
    assert np.all((q == 0) | (q >= 20))                     # MOQ
    assert np.all((q * 1.0).sum(axis=1) <= 40 + 1e-6)       # budget per week


def test_no_order_when_stock_is_enough():
    P = params()
    res = solve_plan(problem(P, np.full((1, 4, 2), 5.0), on_hand=np.array([100.0, 100.0])))
    assert res.orders.sum() == 0


def test_lead_time_and_fixed_cost_batching():
    """With a high fixed cost the model batches demand into fewer, larger orders."""
    P = params(n=1, K=50.0)
    D = np.full((1, 6, 1), 20.0)
    cheap = solve_plan(problem(params(n=1, K=0.1), D, on_hand=np.array([20.0])))
    dear = solve_plan(problem(P, D, on_hand=np.array([20.0])))
    assert (dear.orders > 0).sum() < (cheap.orders > 0).sum()
    # nothing ordered in the last week: it could not arrive inside the horizon (L = 1)
    assert dear.orders[-1].sum() == 0


def test_scenarios_buy_more_than_median_when_shortage_is_expensive():
    P = params(n=1, K=1.0)
    rng = np.random.default_rng(0)
    D = rng.lognormal(np.log(20), 0.6, size=(20, 4, 1))
    point = np.median(D, axis=0, keepdims=True)
    q_point = solve_plan(problem(P, point)).orders.sum()
    q_scen = solve_plan(problem(P, D)).orders.sum()
    assert q_scen >= q_point


def test_evaluate_plan_accounting():
    P = params(n=1, K=10.0)
    orders = np.array([[30.0], [0.0], [0.0]])
    D = np.array([[[10.0], [10.0], [40.0]]])
    ev = evaluate_plan(orders, P, np.array([10.0]), np.zeros((1, 2)), D, capacity=1e6, overflow_cost=1.0)
    # week0: stock 10, sell 10. week1: +30 arrive, sell 10 -> 20. week2: sell 20, lose 20
    assert ev["  Ordering"] == pytest.approx(10.0)
    assert ev["  Lost-sale penalty"] == pytest.approx(20 * 1.2)
    assert ev["  Holding"] == pytest.approx(20 * 0.01)
