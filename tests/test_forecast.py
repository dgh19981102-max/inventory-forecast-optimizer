import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from invopt.forecast import (ETSForecaster, ErrorLibrary, accuracy, clean_history, moving_average,
                             rolling_forecasts, seasonal_naive)


def synthetic(T=90, N=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    base = np.array([100, 50, 20])[None, :N]
    Y = base * (1 + 0.3 * np.sin(2 * np.pi * t / 52)) + rng.normal(0, 5, (T, N))
    return np.clip(Y, 0, None).round()


def test_baselines():
    Y = np.arange(60, dtype=float)[:, None].repeat(2, axis=1)
    assert np.allclose(seasonal_naive(Y, 55, 2)[:, 0], [4, 5])            # same week last year
    assert np.allclose(moving_average(Y, 55, 3, k=4)[:, 0], 53.5)


def test_clean_history_fills_closed_weeks():
    Y = np.ones((10, 2)) * 10
    Y[4] = 0
    assert np.allclose(clean_history(Y)[4], 10)


def test_fast_ets_matches_statsmodels():
    Y = synthetic()
    fast = ETSForecaster(Y).predict(70, 4)
    ref = np.expm1(ExponentialSmoothing(np.log1p(Y[:71, 0]), trend="add", damped_trend=True,
                                        initialization_method="estimated").fit().forecast(4))
    assert np.allclose(fast[:, 0], ref, rtol=1e-6)


def test_backtest_has_no_lookahead():
    """Changing data AFTER the origin must not change the forecast made at that origin."""
    Y = synthetic()
    weeks = pd.date_range("2020-01-06", periods=len(Y), freq="W-MON")
    a = rolling_forecasts(Y, weeks, [70], H=3, models=["MovingAvg4", "ETS", "LightGBM"])
    Y2 = Y.copy(); Y2[71:] *= 5
    b = rolling_forecasts(Y2, weeks, [70], H=3, models=["MovingAvg4", "ETS", "LightGBM"])
    assert np.allclose(a["forecast"], b["forecast"])


def test_accuracy_and_error_library():
    Y = synthetic()
    weeks = pd.date_range("2020-01-06", periods=len(Y), freq="W-MON")
    bt = rolling_forecasts(Y, weeks, range(55, 89), H=3, skus=["a", "b", "c"], models=["MovingAvg4"])
    acc = accuracy(bt)
    assert 0 < acc.loc["MovingAvg4", "WAPE"] < 0.5
    lib = ErrorLibrary(bt, "MovingAvg4", 3)
    q = lib.quantiles(88)
    assert (q[0] <= q[1]).all() and (q[1] <= q[2]).all()
    sc = lib.scenarios(np.ones((3, 3)) * 10, 88, 5, np.random.default_rng(0), skus=["a", "b", "c"])
    assert sc.shape == (5, 3, 3) and (sc >= 0).all()
    # only origins whose targets are already observed are used
    assert max(k for k in lib.joint if k + 3 <= 70) <= 67
