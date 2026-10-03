"""Demand forecasting: baselines, ETS, LightGBM, rolling-origin backtest and prediction intervals.

Conventions
-----------
* ``Y`` is a 2-D array (weeks x SKUs) of weekly demand.
* An *origin* ``o`` is the index of the last week we have observed. A forecast made at
  origin ``o`` for horizon ``h`` (1..H) predicts week ``o + h``.
* Every model only touches ``Y[: o + 1]`` -> no look-ahead.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON = 52
MODELS = ["SeasonalNaive", "MovingAvg4", "ETS", "LightGBM", "Ensemble"]


# --------------------------------------------------------------------------- helpers
def closure_mask(Y: np.ndarray) -> np.ndarray:
    """Weeks where the whole shop sold nothing (Christmas closure)."""
    return Y.sum(axis=1) == 0


def clean_history(Y: np.ndarray) -> np.ndarray:
    """Replace shop-closure weeks by the average of the surrounding open weeks.

    The shop knows its own closure calendar, so a model should not learn "demand collapses
    at Christmas" from a week where the doors were simply shut.
    """
    Y = Y.astype(float).copy()
    closed = closure_mask(Y)
    open_idx = np.where(~closed)[0]
    for t in np.where(closed)[0]:
        before = open_idx[open_idx < t][-2:]
        after = open_idx[open_idx > t][:2]
        nb = np.concatenate([before, after])
        Y[t] = Y[nb].mean(axis=0)
    return Y


# --------------------------------------------------------------------------- simple models
def seasonal_naive(Y: np.ndarray, o: int, H: int) -> np.ndarray:
    """Same week last year; falls back to last value if not enough history."""
    out = np.empty((H, Y.shape[1]))
    for h in range(1, H + 1):
        t = o + h - SEASON
        out[h - 1] = Y[t] if t >= 0 else Y[o]
    return out


def moving_average(Y: np.ndarray, o: int, H: int, k: int = 4) -> np.ndarray:
    return np.repeat(Y[max(0, o - k + 1): o + 1].mean(axis=0, keepdims=True), H, axis=0)


class ETSForecaster:
    """Holt's linear method with damped trend, per SKU, on log(1 + demand).

    * log scale: weekly demand is right-skewed (occasional bulk orders); on the log scale a
      single big order does not drag the level up for weeks.
    * no 52-week seasonality: with ~2 years of weekly data a seasonal ETS cannot be estimated
      reliably, so seasonality is left to the LightGBM model.
    * speed: smoothing parameters are re-estimated (statsmodels, max. likelihood) every
      `refit_every` origins; in between, the same parameters are run through the data up to
      the new origin (the standard Holt recursion below), which takes microseconds.

        level_t = a*y_t + (1-a)*(level_{t-1} + phi*trend_{t-1})
        trend_t = b*(level_t - level_{t-1}) + (1-b)*phi*trend_{t-1}
        forecast_{t+h} = level_t + (phi + phi^2 + ... + phi^h) * trend_t
    """

    def __init__(self, Y: np.ndarray, refit_every: int = 4):
        self.Z = np.log1p(Y)
        self.refit_every = refit_every
        self.params = None          # (alpha, beta, phi, l0, b0) arrays of length N
        self.fitted_at = None

    def _estimate(self, o: int):
        N = self.Z.shape[1]
        prm = np.zeros((5, N))
        for j in range(N):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(self.Z[: o + 1, j], trend="add", damped_trend=True,
                                           initialization_method="estimated").fit()
            pp = fit.params
            prm[:, j] = [pp["smoothing_level"], pp["smoothing_trend"], pp["damping_trend"],
                         pp["initial_level"], pp["initial_trend"]]
        self.params, self.fitted_at = prm, o

    def predict(self, o: int, H: int) -> np.ndarray:
        if self.fitted_at is None or o - self.fitted_at >= self.refit_every or o < self.fitted_at:
            self._estimate(o)
        a, b, phi, lev, tr = (x.copy() for x in self.params)
        for t in range(o + 1):
            prev = lev
            lev = a * self.Z[t] + (1 - a) * (lev + phi * tr)
            tr = b * (lev - prev) + (1 - b) * phi * tr
        damp = np.cumsum(phi[None, :] ** np.arange(1, H + 1)[:, None], axis=0)   # (H, N)
        return np.clip(np.expm1(lev[None] + damp * tr[None]), 0, None)


def ets(Y: np.ndarray, o: int, H: int) -> np.ndarray:
    """One-off ETS forecast (parameters estimated at this origin)."""
    return ETSForecaster(Y).predict(o, H)


# --------------------------------------------------------------------------- LightGBM (global, direct)
# objective='l1' (median): robust to bulk-order spikes and minimises WAPE.
LGB_PARAMS = dict(objective="l1", n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  verbose=-1, n_jobs=4)


def _features(Y: np.ndarray, weeks: pd.DatetimeIndex, h: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature tensor X[o, sku, f] for horizon h and normalised target y[o, sku].

    All features at origin o use Y[: o+1] only, except the target (Y[o+h]).
    Everything is divided by the SKU's recent level (13-week mean) so one global model
    can learn across products of very different volume.
    """
    T, N = Y.shape
    nan = np.full((T, N), np.nan)

    def shift(a, k):  # value k weeks before (k>0) / after (k<0)
        out = nan.copy()
        if k >= 0:
            out[k:] = a[: T - k]
        else:
            out[: T + k] = a[-k:]
        return out

    csum = np.vstack([np.zeros((1, N)), np.cumsum(Y, axis=0)])

    def rmean(k):  # mean of last k weeks ending at o (inclusive)
        out = nan.copy()
        out[k - 1:] = (csum[k:] - csum[:-k]) / k
        return out

    scale = rmean(13) + 1.0
    feats, names = [], []
    for k in range(1, 9):
        feats.append(shift(Y, k - 1) / scale); names.append(f"lag{k}")
    for k in (4, 8):
        feats.append(rmean(k) / scale); names.append(f"rmean{k}")
    sq = rmean(13)
    var = nan.copy()
    c2 = np.vstack([np.zeros((1, N)), np.cumsum(Y ** 2, axis=0)])
    var[12:] = (c2[13:] - c2[:-13]) / 13 - sq[12:] ** 2
    feats.append(np.sqrt(np.clip(var, 0, None)) / scale); names.append("std13")
    ly_target = shift(Y, SEASON - h)                    # Y[o + h - 52]
    ly_level = shift(rmean(13), SEASON)                 # level a year before the origin
    feats.append(ly_target / scale); names.append("last_year_same_week")
    feats.append(ly_target / (ly_level + 1.0)); names.append("last_year_seasonal_ratio")
    target_week = (weeks + pd.Timedelta(weeks=h)).isocalendar().week.to_numpy().astype(float)
    feats.append(np.repeat(target_week[:, None], N, axis=1)); names.append("week_of_year")
    X = np.stack(feats, axis=2)
    y = shift(Y, -h) / scale
    return X, y, names, scale


class LightGBMForecaster:
    """Direct multi-horizon global model: one LightGBM regressor per horizon h."""

    def __init__(self, Y: np.ndarray, weeks: pd.DatetimeIndex, H: int, retrain_every: int = 4,
                 min_train_origin: int = 13):
        self.Y, self.H, self.retrain_every, self.min_o = Y, H, retrain_every, min_train_origin
        self.cache = {h: _features(Y, weeks, h) for h in range(1, H + 1)}
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self.trained_at: int | None = None

    def _train(self, o: int):
        for h in range(1, self.H + 1):
            X, y, _, _ = self.cache[h]
            rows = np.arange(self.min_o, o - h + 1)          # target o'+h <= o  -> known at o
            Xr = X[rows].reshape(-1, X.shape[2]); yr = y[rows].reshape(-1)
            ok = ~np.isnan(yr)
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=h)
            m.fit(Xr[ok], yr[ok])
            self.models[h] = m
        self.trained_at = o

    def predict(self, o: int) -> np.ndarray:
        if self.trained_at is None or o - self.trained_at >= self.retrain_every or o < self.trained_at:
            self._train(o)
        out = np.empty((self.H, self.Y.shape[1]))
        for h in range(1, self.H + 1):
            X, _, _, scale = self.cache[h]
            out[h - 1] = self.models[h].predict(X[o]) * scale[o]
        return np.clip(out, 0, None)


# --------------------------------------------------------------------------- backtest
def rolling_forecasts(Y_raw: np.ndarray, weeks: pd.DatetimeIndex, origins, H: int = 6,
                      skus: list[str] | None = None, models=MODELS, retrain_every: int = 4) -> pd.DataFrame:
    """Rolling-origin forecasts for every model. Returns a long table
    [origin, week, h, sku, model, forecast, actual]."""
    Y = clean_history(Y_raw)
    N = Y.shape[1]
    skus = skus or [str(i) for i in range(N)]
    lgbf = LightGBMForecaster(Y, weeks, H, retrain_every) if {"LightGBM", "Ensemble"} & set(models) else None
    etsf = ETSForecaster(Y, refit_every=retrain_every)
    rows = []
    for o in origins:
        f = {}
        if "SeasonalNaive" in models: f["SeasonalNaive"] = seasonal_naive(Y, o, H)
        if "MovingAvg4" in models: f["MovingAvg4"] = moving_average(Y, o, H, 4)
        if {"ETS", "Ensemble"} & set(models): f["ETS"] = etsf.predict(o, H)
        if lgbf is not None: f["LightGBM"] = lgbf.predict(o)
        if "Ensemble" in models: f["Ensemble"] = 0.5 * (f["ETS"] + f["LightGBM"])
        for m in models:
            for h in range(1, H + 1):
                t = o + h
                actual = Y_raw[t] if t < len(Y_raw) else np.full(N, np.nan)
                rows.append(pd.DataFrame({"origin": o, "week": weeks[o] + pd.Timedelta(weeks=h), "h": h,
                                          "sku": skus, "model": m, "forecast": f[m][h - 1], "actual": actual}))
    df = pd.concat(rows, ignore_index=True)
    closed = closure_mask(Y_raw)
    df["closed_week"] = [bool(closed[o + h]) if o + h < len(closed) else False for o, h in zip(df.origin, df.h)]
    return df


def accuracy(df: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """WAPE, MAPE and bias. Shop-closure weeks are excluded (the shop knows it will be closed)."""
    d = df.dropna(subset=["actual"])
    d = d[~d["closed_week"]]

    def agg(g):
        e = g["forecast"] - g["actual"]
        pos = g["actual"] > 0
        return pd.Series({
            "WAPE": e.abs().sum() / g["actual"].sum(),
            "MAPE": (e[pos].abs() / g["actual"][pos]).mean(),
            "Bias": e.sum() / g["actual"].sum(),
            "n": len(g),
        })
    return d.groupby(list(by)).apply(agg, include_groups=False)


# --------------------------------------------------------------------------- uncertainty
class ErrorLibrary:
    """Empirical forecast-error paths from past backtests -> intervals and demand scenarios.

    For each past forecast origin o' and SKU we store the ratio path r_h = actual / forecast,
    h = 1..H (how wrong the forecast made in week o' turned out to be, 1..H weeks ahead).
    * Intervals: quantiles of r_h (pooled over SKUs) x the point forecast.
    * Scenarios: one scenario = the error matrix of ONE past origin, applied to all SKUs at
      once. This keeps both the week-to-week correlation of errors and the correlation
      between products (busy weeks are busy for everyone), which matters for the shared
      warehouse and budget.
    Only origins whose last target week is <= the current origin are used (no look-ahead).
    """

    def __init__(self, bt: pd.DataFrame, model: str, H: int, floor: float = 1.0, clip: float = 4.0):
        d = bt[(bt["model"] == model) & (~bt["closed_week"])].dropna(subset=["actual"])
        d = d.assign(ratio=(d["actual"] / d["forecast"].clip(lower=floor)).clip(upper=clip))
        wide = d.pivot_table(index=["origin", "sku"], columns="h", values="ratio").dropna()
        self.H = H
        self.origins = wide.index.get_level_values("origin").to_numpy()
        self.path_sku = wide.index.get_level_values("sku").to_numpy()
        self.paths = wide[list(range(1, H + 1))].to_numpy()
        # (origin -> (H, N) ratio matrix) for joint scenarios
        self.skus = sorted(wide.index.get_level_values("sku").unique())
        self.joint = {}
        for o, g in wide.groupby(level="origin"):
            g = g.droplevel("origin")
            if len(g) == len(self.skus):
                self.joint[o] = g.loc[self.skus, list(range(1, H + 1))].to_numpy().T

    def available(self, o: int, sku: str | None = None) -> np.ndarray:
        ok = self.origins + self.H <= o
        if sku is not None:
            ok &= self.path_sku == sku
        return self.paths[ok]

    def quantiles(self, o: int, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
        """Array (len(qs), H) of ratio quantiles."""
        return np.quantile(self.available(o), qs, axis=0)

    def scenarios(self, point: np.ndarray, o: int, n: int, rng: np.random.Generator,
                  skus: list[str] | None = None, pick: str = "recent") -> np.ndarray:
        """Demand scenarios, array (n, H, N) = point forecast x error matrix of a past origin.

        Uses the n most recent eligible past origins (recent errors are most representative).
        """
        H, N = point.shape
        eligible = sorted(k for k in self.joint if k + self.H <= o)
        if pick == "random" and len(eligible) > n:
            past = sorted(rng.choice(eligible, size=n, replace=False))
        else:
            past = eligible[-n:]
        if not past:
            raise ValueError("No past forecast errors available before this origin")
        col = [self.skus.index(s) for s in skus] if skus is not None else list(range(N))
        R = np.stack([self.joint[k][:H][:, col] for k in past])          # (n, H, N)
        return point[None] * R


# Discretised error distribution: finer steps in the upper tail, where stock-out risk lives.
TAIL_BUCKETS = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, .997, 1.0])


def quantile_scenarios(lib: "ErrorLibrary", point: np.ndarray, o: int, skus: list[str],
                       buckets: np.ndarray = TAIL_BUCKETS, errors: str = "pooled",
                       center: str | None = "median", n_lhs: int = 0,
                       rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted demand scenarios from the quantiles of past forecast errors.

    Bucket k covers error quantile levels [b_k, b_k+1]; its scenario multiplies the point forecast
    by the error ratio (actual / forecast) at the bucket mid-point, for every week of the horizon
    (errors are persistent: a forecast that is too low this week tends to stay too low). Its
    probability is the bucket width. 14 weighted scenarios reach the 99.8% level, where 14 equally
    likely random draws would rarely see past 95%.

    errors="pooled": one error distribution for all SKUs (hundreds of past errors -> stable tails).
    errors="sku":    each SKU's own errors (only 15-40 per SKU -> noisy, too-thin tails).
    center="median": divide by the median error ratio, so the middle scenario equals the point
                     forecast but the right skew of demand is kept. This drops the past average
                     bias, which did not persist (forecasts ran high in H1 2011, low in the Q4 ramp-up).
    These choices were made with the calibration check in reports/scenario_calibration.csv.
    Returns (scenarios (K, H, N), probabilities (K,)).
    """
    if n_lhs:
        # Latin-hypercube variant: n equally likely levels, shuffled independently per SKU, so
        # products are NOT all high or all low together (shared warehouse/budget see realistic
        # aggregate variation)
        buckets = np.linspace(0, 1, n_lhs + 1)
    mids = (buckets[:-1] + buckets[1:]) / 2
    probs = np.diff(buckets)
    H, N = point.shape

    def ratio_quantiles(paths):
        q = np.quantile(paths, mids, axis=0)[:, :H]
        if center == "median":
            q = q / np.median(paths, axis=0)[:H]
        elif center == "mean":
            q = q / paths.mean(axis=0)[:H]
        return q

    R = np.empty((len(mids), H, N))
    pooled = ratio_quantiles(lib.available(o))
    for j, sku in enumerate(skus):
        own = lib.available(o, sku) if errors == "sku" else None
        R[:, :, j] = ratio_quantiles(own) if own is not None and len(own) >= 8 else pooled
        if n_lhs:
            R[:, :, j] = R[(rng or np.random.default_rng(o)).permutation(len(mids)), :, j]
    return point[None] * R, probs


def interval_coverage(bt: pd.DataFrame, model: str, H: int, eval_origins, lo=0.1, hi=0.9) -> pd.DataFrame:
    """Share of actuals inside the [P10, P90] band, using only past errors at each origin."""
    lib = ErrorLibrary(bt, model, H)
    d = bt[(bt["model"] == model) & bt["origin"].isin(eval_origins) & ~bt["closed_week"]].dropna(subset=["actual"])
    rows = []
    for o, g in d.groupby("origin"):
        q = lib.quantiles(o, (lo, hi))
        lo_b = g["forecast"].to_numpy() * q[0, g["h"].to_numpy() - 1]
        hi_b = g["forecast"].to_numpy() * q[1, g["h"].to_numpy() - 1]
        rows.append(pd.DataFrame({"h": g["h"], "inside": (g["actual"] >= lo_b) & (g["actual"] <= hi_b)}))
    r = pd.concat(rows)
    return r.groupby("h")["inside"].mean().to_frame("coverage_P10_P90")
