"""Interactive replenishment planner.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invopt.excel_io import read_input, write_output  # noqa: E402
from invopt.forecast import rolling_forecasts  # noqa: E402
from invopt.planner import make_plan  # noqa: E402

st.set_page_config(page_title="Replenishment planner", layout="wide")
EXAMPLE = ROOT / "examples" / "input_template.xlsx"
COLORS = {"Rule of thumb": "#9a9890", "Optimised: point forecast + safety stock": "#2a78d6",
          "Optimised: scenarios (recommended)": "#eb6834"}


@st.cache_data(show_spinner=False)
def load(file_bytes: bytes):
    return read_input(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def backtest(file_bytes: bytes, H: int, fast: bool):
    """The slow part (model backtest) only depends on the history, not on the sliders."""
    inp = load(file_bytes)
    Y = inp.history.to_numpy()
    o_now = len(Y) - 1
    n_bt = 26
    models = ["MovingAvg4", "ETS"] if fast else ["MovingAvg4", "ETS", "LightGBM", "Ensemble"]
    return rolling_forecasts(Y, inp.history.index, range(max(13, o_now - H - n_bt + 1), o_now + 1),
                             H=H, skus=list(inp.products.index), models=models)


st.title("Demand forecast & replenishment planner")
st.caption("Excel in, optimal order plan out. Forecasts with uncertainty -> mixed-integer optimisation "
           "under warehouse, budget, pack-size, MOQ and lead-time constraints.")

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Input workbook (.xlsx)", type="xlsx")
    file_bytes = up.getvalue() if up else EXAMPLE.read_bytes()
    st.caption("Using: " + (up.name if up else "example (UCI Online Retail II, 30 products)"))
    st.download_button("Download blank template", EXAMPLE.read_bytes(), "input_template.xlsx")
    inp = load(file_bytes)
    s0 = inp.settings

    st.header("2. What-if")
    b0, c0 = float(s0["weekly_budget"]), float(s0["warehouse_capacity_space"])
    budget = st.slider("Weekly purchase budget", 0.3 * b0, 2.0 * b0, b0, step=b0 / 20, format="%.0f")
    cap = st.slider("Warehouse capacity (space units)", 0.5 * c0, 2.0 * c0, c0, step=c0 / 20, format="%.0f")
    svc = st.slider("Service-level target (safety-stock method)", 0.80, 0.99, float(s0["service_level_target"]), 0.01)
    k = st.slider("Fixed cost per order line", 0.0, 50.0, float(inp.products["fixed_order_cost"].median()), 1.0)
    fast = st.checkbox("Fast mode (skip LightGBM in model selection)", value=False)
    go_btn = st.button("Optimise", type="primary", width="stretch")

if go_btn or "res" not in st.session_state:
    H = int(s0["planning_horizon_weeks"])
    with st.spinner("Backtesting forecast models on your history (cached after the first run) ..."):
        bt = backtest(file_bytes, H, fast)
    inp2 = load(file_bytes)
    inp2.products = inp2.products.assign(fixed_order_cost=k)
    with st.spinner("Solving the order-plan MILP ..."):
        st.session_state.res = make_plan(inp2, {"weekly_budget": budget, "warehouse_capacity_space": cap,
                                                "service_level_target": svc}, backtest=bt)
res = st.session_state.res
cmp_ = res["comparison"]
rec = "Optimised: scenarios (recommended)"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected cost, recommended plan", f"{cmp_.loc['Expected total cost', rec]:,.0f}",
          delta=f"-{cmp_.loc['Saving vs rule of thumb', rec]} vs rule of thumb", delta_color="off")
c2.metric("Expected fill rate", f"{cmp_.loc['Expected fill rate', rec]:.1%}",
          delta=f"rule of thumb {cmp_.loc['Expected fill rate', 'Rule of thumb']:.1%}", delta_color="off")
c3.metric("Order lines this week", int((res["orders"][0] > 0).sum()))
c4.metric("Forecast model / solve time", res["model"], delta=f"MILP {res['solve_seconds']:.1f}s", delta_color="off")

tab1, tab2, tab3 = st.tabs(["Order plan", "Product view", "Compare plans"])
with tab1:
    P = res["products"]
    plan = pd.DataFrame(res["orders"].T.astype(int), index=P.index,
                        columns=[f"W{t+1} {w:%d %b}" for t, w in enumerate(res["weeks"])])
    plan.insert(0, "product", P["description"])
    st.caption("Week 1 = orders to place now. Re-run next week with updated stock (rolling horizon).")
    st.dataframe(plan, width="stretch", height=480)
    buf = io.BytesIO()
    write_output(buf, res)
    st.download_button("Download plan as Excel", buf.getvalue(), "replenishment_plan.xlsx", type="primary")

with tab2:
    P = res["products"]
    sku = st.selectbox("Product", list(P.index), format_func=lambda s: f"{s} - {P.loc[s, 'description']}")
    i = list(P.index).index(sku)
    hist = inp.history[sku].iloc[-26:]
    det = res["detail"][res["detail"]["sku"] == sku]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Demand (history)", line=dict(color="#0b0b0b", width=2)))
    fig.add_trace(go.Scatter(x=list(det["week_start"]) + list(det["week_start"])[::-1],
                             y=list(det["forecast_P90"]) + list(det["forecast_P10"])[::-1], fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=det["week_start"], y=det["forecast_P50"], name="Forecast", line=dict(color="#2a78d6", width=2)))
    fig.add_trace(go.Bar(x=det["week_start"], y=det["order_qty"], name="Order placed", marker_color="#eb6834", width=3 * 86400000))
    fig.add_trace(go.Scatter(x=det["week_start"], y=det["projected_stock_end_of_week"], name="Projected stock",
                             line=dict(color="#1baf7a", width=2, dash="dot")))
    fig.update_layout(height=420, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1), yaxis_title="Units")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(det.drop(columns=["sku", "description"]).set_index("week_start"), width="stretch")

with tab3:
    st.caption(f"All three order plans scored on the same {res['n_eval_scenarios']} demand scenarios "
               "built from this history's own past forecast errors.")
    cost_rows = ["  Ordering", "  Holding", "  Lost-sale penalty", "  Overflow space"]
    fig = go.Figure()
    for r_, col in zip(cost_rows, ["#2a78d6", "#1baf7a", "#eb6834", "#eda100"]):
        fig.add_trace(go.Bar(y=cmp_.columns, x=cmp_.loc[r_].astype(float), name=r_.strip(), orientation="h",
                             marker=dict(color=col, line=dict(color="#fcfcfb", width=2))))
    fig.update_layout(barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.25), xaxis_title="Expected cost over the horizon")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cmp_.astype(str), width="stretch")
    st.markdown("**Forecast backtest on your data** (WAPE: lower is better)")
    st.dataframe(res["accuracy"].round(3), width="stretch")
