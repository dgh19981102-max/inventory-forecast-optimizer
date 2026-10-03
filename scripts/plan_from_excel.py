"""Excel in -> replenishment plan out.

    python scripts/plan_from_excel.py examples/input_template.xlsx examples/output_plan.xlsx
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invopt.excel_io import read_input, write_output  # noqa: E402
from invopt.planner import make_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input workbook (same format as examples/input_template.xlsx)")
    ap.add_argument("output", help="output workbook to write")
    ap.add_argument("--fast", action="store_true", help="skip LightGBM in the model backtest (faster)")
    a = ap.parse_args()
    t = time.time()
    inp = read_input(a.input)
    models = ("MovingAvg4", "ETS") if a.fast else ("MovingAvg4", "ETS", "LightGBM", "Ensemble")
    res = make_plan(inp, models=models, progress=print)
    write_output(a.output, res)
    print(res["comparison"].to_string())
    print(f"\nwrote {a.output}  ({time.time() - t:.0f}s total, optimiser {res['solve_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
