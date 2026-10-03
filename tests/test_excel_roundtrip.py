from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invopt.excel_io import read_input, write_output, write_template
from invopt.planner import make_plan


@pytest.fixture
def workbook(tmp_path):
    rng = np.random.default_rng(1)
    weeks = pd.date_range("2022-01-03", periods=80, freq="W-MON")
    hist = pd.DataFrame(rng.poisson([60, 30, 15], size=(80, 3)), index=weeks, columns=["A1", "B2", "C3"]).astype(float)
    prod = pd.DataFrame({
        "description": ["alpha", "beta", "gamma"], "sell_price": [2.0, 3.0, 5.0], "unit_cost": [1.0, 1.5, 2.5],
        "space_per_unit": [1.0, 1.0, 2.0], "holding_cost_week": [0.01, 0.015, 0.03],
        "lost_sale_penalty": [1.2, 1.8, 3.0], "case_size": [6, 6, 2], "moq": [12, 12, 4],
        "lead_time_weeks": [1, 2, 1], "fixed_order_cost": 10.0, "on_hand": [60, 40, 20],
        "on_order_week1": [30, 30, 0], "on_order_week2": [0, 30, 0],
    }, index=pd.Index(["A1", "B2", "C3"], name="sku"))
    settings = {"warehouse_capacity_space": 400.0, "weekly_budget": 300.0, "overflow_cost_per_space_week": 0.05,
                "planning_horizon_weeks": 4, "service_level_target": 0.95, "random_seed": 1}
    path = tmp_path / "in.xlsx"
    write_template(path, settings, prod, hist)
    return path


def test_roundtrip_and_plan(workbook, tmp_path):
    inp = read_input(workbook)
    assert list(inp.products.index) == ["A1", "B2", "C3"]
    assert inp.history.shape == (80, 3)
    res = make_plan(inp, models=("MovingAvg4", "ETS"))
    assert res["orders"].shape == (4, 3)
    assert (res["orders"] >= 0).all()
    out = tmp_path / "out.xlsx"
    write_output(out, res)
    sheets = pd.ExcelFile(out).sheet_names
    assert {"Summary", "Order Plan", "Plan Detail", "Forecast Accuracy", "Assumptions"} <= set(sheets)
