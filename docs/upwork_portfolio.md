# Upwork portfolio entry (hand this to the Upwork conversation)

## Title (≤ 70 characters)
Demand Forecasting & Inventory Optimization – Excel In, Reorder Plan Out

## Role
Python developer, optimization modeling (solo project)

## Description

**39% lower inventory cost and 66% fewer stock-outs than "reorder last month's sales", measured by replaying 27 weeks of real retail demand (under stated cost assumptions).**

**Problem:** A small retailer with 30 products, a limited warehouse, a weekly cash budget, supplier pack sizes, minimum order quantities and 1–2-week lead times. They need to decide every week what to reorder, and how much.

**Approach:**
- Forecasted weekly demand with 5 models (moving average, seasonal naive, ETS, LightGBM, ensemble), compared in a 47-week rolling backtest. Calibrated 80% prediction intervals: 82–84% coverage.
- Turned past forecast errors into weighted demand scenarios.
- Built a mixed-integer optimization model (OR-Tools + open-source SCIP). It picks order quantities under capacity, budget, pack-size, MOQ and lead-time constraints, and minimises expected ordering + holding + lost-sale cost. Each plan solves in seconds.
- Simulated three policies week by week on the real 2011 demand: rule of thumb, point forecast + safety stock, uncertainty-aware.

**Results:**
- 39% lower total relevant cost than the rule of thumb (£16.8k → £10.3k over 27 weeks).
- Stock-out SKU-weeks down 66%, lost units down 56%, order lines down 61%.
- 20% cheaper even than the rule of thumb tuned in hindsight; 42% cheaper on an untouched hold-out period.
- Clients only need Excel: fill in a template, get back an order plan plus a cost comparison. An optional Streamlit app adds what-if sliders.

Public data: UCI Online Retail II (CC BY 4.0). All cost assumptions are documented and editable.

## Skills (tags)
Mathematical Optimization · Operations Research · Python · Forecasting · Supply Chain · Microsoft Excel

(extra if allowed: Inventory Management · Linear Programming · Time Series Analysis · Streamlit)

## Images to upload (in this order)
1. `reports/figures/policy_cost_comparison.png` – cover image (the headline result)
2. `reports/figures/app_order_plan.png` – the tool itself
3. `reports/figures/app_product_view.png` – forecast + interval + planned orders
4. `reports/figures/inventory_trajectory.png` – what the policies do differently

## Project link
https://github.com/dgh19981102-max/inventory-forecast-optimizer

## Fixed-price catalog idea (Project Catalog)
**"I will build an inventory / replenishment optimization model from your Excel data" – from $150**

| Tier | Price | What's included |
|---|---|---|
| Basic | $150 | Up to 30 SKUs: demand forecast with intervals and a 4–6 week optimal order plan in Excel, with a 1-page summary of savings vs your current rule |
| Standard | $400 | Up to 200 SKUs plus your real constraints (warehouse, budget, MOQ, lead times), a backtest on your history and a cost-comparison report |
| Premium | $900 | Standard + a Streamlit what-if app or reusable Python script, a handover call and 2 weeks of adjustments |

Requirements from the buyer: 1–2 years of weekly or daily sales per product, unit costs, pack sizes / MOQ, lead times, current stock.
