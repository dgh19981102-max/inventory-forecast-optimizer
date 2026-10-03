"""Step 3 - replay real demand with three replenishment policies.

    python scripts/run_simulation.py                 # main test Jun-Nov 2011 + hold-out Mar-May 2011 (~15 min)
    python scripts/run_simulation.py --sensitivity   # + fixed-order-cost sensitivity (~20 min more)
    python scripts/run_simulation.py --ablation      # + the scenario-design variants that were tried (~45 min)
"""
import argparse
import functools
import time

import numpy as np
import pandas as pd
from _common import FIGS, H, PROCESSED, REPORTS, SIM_END, SIM_START, load_processed

import invopt.simulate as simmod
from invopt import figures
from invopt.forecast import ErrorLibrary, accuracy, quantile_scenarios
from invopt.params import Assumptions, assumptions_table, build_sku_params, global_settings
from invopt.simulate import OptimisedPolicy, RuleOfThumb, simulate, summarise

HOLDOUT = (66, 77)    # 2011-03-07 .. 2011-05-23: never used while designing the method


def setup(weekly, master, bt, t0, a: Assumptions | None = None):
    ref = slice(0, t0)                                  # parameters sized on data before the test starts
    P = build_sku_params(master, weekly, a, reference_weeks=ref)
    S = global_settings(P, weekly, a, reference_weeks=ref)
    # forecasting model chosen ONLY from backtest errors observed before the test starts
    model = accuracy(bt[bt["origin"] + H <= t0 - 1])["WAPE"].idxmin()
    return P, S, model


def run_policies(Y, weeks, P, S, bt, model, t0, t1, label="", verbose=False):
    lib = ErrorLibrary(bt, model, H)
    pols = [RuleOfThumb(review=1, buffer_weeks=1.0),
            OptimisedPolicy(bt, lib, model, scenarios=False),
            OptimisedPolicy(bt, lib, model, scenarios=True, verbose=verbose)]
    out, logs = {}, {}
    for pol in pols:
        t = time.time()
        log, wk = simulate(pol, Y, P, S, t0, t1, weeks)
        out[pol.name] = summarise(log, wk)
        logs[pol.name] = log
        st = getattr(pol, "solve_times", [])
        msg = f" (MILP: mean {np.mean(st):.1f}s, max {np.max(st):.1f}s per weekly plan)" if st else ""
        print(f"{label}{pol.name}: {time.time() - t:.0f}s{msg}", flush=True)
    return pd.DataFrame(out), logs


def add_relative_rows(summary):
    base = summary.iloc[:, 0]
    summary.loc["Cost saving vs rule of thumb"] = 1 - summary.loc["Total relevant cost (£)"] / base["Total relevant cost (£)"]
    summary.loc["Lost units reduction vs rule of thumb"] = 1 - summary.loc["Lost units"] / base["Lost units"]
    summary.loc["Stock-out SKU-weeks reduction vs rule of thumb"] = 1 - summary.loc["Stock-out SKU-weeks"] / base["Stock-out SKU-weeks"]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--skip-main", action="store_true")
    a = ap.parse_args()
    weekly, master = load_processed()
    Y = weekly.to_numpy()
    bt = pd.read_parquet(PROCESSED / "backtest.parquet")
    P, S, model = setup(weekly, master, bt, SIM_START)
    print(f"forecast model used by the optimiser: {model}\nsettings: {S}")

    if not a.skip_main:
        P.to_csv(REPORTS / "sku_parameters.csv")
        pd.concat([assumptions_table(), pd.Series(S, name="value").to_frame()]).to_csv(REPORTS / "assumptions.csv")
        summary, logs = run_policies(Y, weekly.index, P, S, bt, model, SIM_START, SIM_END, verbose=True)
        pd.concat(logs, names=["policy"]).to_parquet(PROCESSED / "simulation_logs.parquet")

        # fairness check: the best rule-of-thumb settings *in hindsight* (tuned on the test period itself)
        grid = {}
        for R in (1, 2, 4):
            for b in (0, 0.5, 1, 1.5, 2, 2.5, 3):
                log, wk = simulate(RuleOfThumb(R, b), Y, P, S, SIM_START, SIM_END, weekly.index)
                grid[f"review {R}w, buffer {b}w"] = summarise(log, wk)
        grid = pd.DataFrame(grid).T.sort_values("Total relevant cost (£)")
        grid.round(3).to_csv(REPORTS / "rule_of_thumb_grid.csv")
        summary.insert(1, f"Rule of thumb, best in hindsight ({grid.index[0]})", grid.iloc[0])
        summary = add_relative_rows(summary)
        summary.round(3).to_csv(REPORTS / "policy_comparison.csv")
        print("\n", summary.round(3).to_string(), flush=True)
        figures.policy_cost_breakdown(summary[[c for c in summary.columns if "hindsight" not in c]],
                                      FIGS / "policy_cost_comparison.png")
        sku = "85123A"
        figures.inventory_trajectory(logs, sku, master.loc[sku, "description"].strip(), FIGS / "inventory_trajectory.png")

        # hold-out window: same code, earlier period, parameters/model chosen with data before it
        P2, S2, model2 = setup(weekly, master, bt, HOLDOUT[0])
        print(f"\nHold-out {weekly.index[HOLDOUT[0]].date()}..{weekly.index[HOLDOUT[1]].date()}, model {model2}")
        hold, _ = run_policies(Y, weekly.index, P2, S2, bt, model2, *HOLDOUT, label="[hold-out] ")
        hold = add_relative_rows(hold)
        hold.round(3).to_csv(REPORTS / "policy_comparison_holdout.csv")
        print("\n", hold.round(3).to_string(), flush=True)

    if a.sensitivity:
        rows = {}
        for K in (5.0, 15.0, 30.0):
            P2, S2, _ = setup(weekly, master, bt, SIM_START, Assumptions(fixed_order_cost=K))
            sm, _ = run_policies(Y, weekly.index, P2, S2, bt, model, SIM_START, SIM_END, label=f"[K={K}] ")
            rows[f"fixed order cost £{K:.0f}"] = sm.loc["Total relevant cost (£)"]
        sens = pd.DataFrame(rows).T
        for c in list(sens.columns[1:]):
            sens[f"saving: {c}"] = 1 - sens[c] / sens.iloc[:, 0]
        sens.round(3).to_csv(REPORTS / "sensitivity_fixed_order_cost.csv")
        print("\n", sens.round(3).to_string(), flush=True)

    if a.ablation:
        # the scenario designs that were tried before settling on the default (see README)
        variants = {
            "own-SKU errors, raw, 14 tail-weighted scenarios": dict(q=dict(errors="sku", center=None)),
            "own-SKU errors, mean-centred": dict(q=dict(errors="sku", center="mean")),
            "pooled, median-centred + recourse": dict(q={}, recourse=True),
            "pooled, median-centred + recourse + per-scenario capacity": dict(q={}, recourse=True, cap=True),
            "pooled, median-centred, 20 Latin-hypercube scenarios + recourse + per-scenario capacity":
                dict(q=dict(n_lhs=20), recourse=True, cap=True),
            "pooled, median-centred (DEFAULT)": dict(q={}),
        }
        import invopt.optimize as opt
        rows = {}
        lib = ErrorLibrary(bt, model, H)
        orig_q, orig_init = simmod.quantile_scenarios, opt.PlanProblem.__init__
        for name, v in variants.items():
            simmod.quantile_scenarios = functools.partial(quantile_scenarios, **v["q"])

            def init(self, *args, _v=v, **kw):
                orig_init(self, *args, **kw)
                self.recourse = _v.get("recourse", False)
                self.capacity_per_scenario = _v.get("cap", False)
            opt.PlanProblem.__init__ = init
            pol = OptimisedPolicy(bt, lib, model, scenarios=True)
            t = time.time()
            log, wk = simulate(pol, Y, P, S, SIM_START, SIM_END, weekly.index)
            rows[name] = summarise(log, wk)
            print(f"[ablation] {name}: {time.time() - t:.0f}s", flush=True)
        simmod.quantile_scenarios, opt.PlanProblem.__init__ = orig_q, orig_init
        abl = pd.DataFrame(rows).T
        abl.round(3).to_csv(REPORTS / "scenario_design_ablation.csv")
        print("\n", abl.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
