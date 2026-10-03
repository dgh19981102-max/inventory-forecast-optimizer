# One-command reproduction:  make all
PY ?= python

all: data backtest simulate excel test

data:        ## raw UCI file -> weekly demand of 30 stable SKUs
	$(PY) scripts/prepare_data.py
backtest:    ## rolling-origin forecast backtest (~2 min)
	$(PY) scripts/run_backtest.py
simulate:    ## replay Jun-Nov 2011 with 3 policies (~10-15 min)
	$(PY) scripts/run_simulation.py
excel:       ## example Excel in -> Excel plan out (~1-2 min)
	$(PY) scripts/make_excel_template.py
	$(PY) scripts/plan_from_excel.py examples/input_template.xlsx examples/output_plan.xlsx
app:
	streamlit run app/streamlit_app.py
test:
	$(PY) -m pytest -q
