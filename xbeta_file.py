"""
Read a CSV containing daily closing prices (first column after Date is
treated as the benchmark) and compute, for every ticker:

  * nobs       - number of daily return observations
  * ann_ret    - geometric annualized return (percent)
  * vol        - annualized volatility (percent)
  * skew       - daily return skewness
  * kurt       - excess kurtosis
  * min, max   - minimum and maximum daily return (percent)
  * corr       - correlation with the benchmark
  * beta       - regression slope to benchmark
  * alpha      - regression intercept, annualized (percent)

ASCII only. Requires numpy, pandas, statsmodels.

Usage:
    python xbeta_from_file.py
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

pd.options.display.float_format = '{:.3f}'.format
# -------------------------------------------------------------------
# constants
# -------------------------------------------------------------------
OBS_YEAR  = 252.0          # trading days per year
RET_SCALE = 100.0          # returns expressed in percent
VOL_ANN   = np.sqrt(OBS_YEAR)

IN_FILE = "prices.csv"

# -------------------------------------------------------------------
# read price data
# -------------------------------------------------------------------
df_prices = pd.read_csv(
    IN_FILE,
    parse_dates=["Date"],
    index_col="Date",
)

if df_prices.empty:
    sys.exit("price file is empty")

tickers   = list(df_prices.columns)
benchmark = tickers[0]

print("file       :", IN_FILE)
print("benchmark  :", benchmark)
print("start date :", df_prices.index.min().date())
print("end date   :", df_prices.index.max().date())

# -------------------------------------------------------------------
# helper functions
# -------------------------------------------------------------------
def annualized_return(prices):
    """Geometric annualized return in percent."""
    if len(prices) < 2:
        return np.nan
    gross = prices.pct_change().dropna() + 1.0
    years = len(gross) / OBS_YEAR
    cagr  = gross.prod() ** (1.0 / years) - 1.0
    return float(cagr * RET_SCALE)


def series_stats(ret_ser):
    """Return vol, skew, kurt, min, max for percent returns."""
    s = ret_ser.squeeze()
    return {
        "vol":  float(VOL_ANN * s.std()),
        "skew": float(s.skew()),
        "kurt": float(s.kurtosis()),
        "min":  float(s.min()),
        "max":  float(s.max()),
    }


def compute_stats(asset, benchmark, df_px):
    """Compute all requested stats for asset relative to benchmark."""
    s_asset = df_px[asset].dropna()
    ann_ret = annualized_return(s_asset)

    if asset == benchmark:
        ret = RET_SCALE * s_asset.pct_change().dropna()
        out = {
            "symbol":  asset,
            "nobs":    len(ret),
            "corr":    1.0,
            "beta":    1.0,
            "alpha":   0.0,
            "ann_ret": ann_ret,
        }
        out.update(series_stats(ret))
        return out

    df_pair = df_px[[asset, benchmark]].dropna()
    df_ret  = RET_SCALE * df_pair.pct_change().dropna()

    y = df_ret[asset]
    x = df_ret[benchmark]

    model = sm.OLS(y, sm.add_constant(x)).fit()
    beta  = float(model.params[benchmark])
    alpha_daily_pct = float(model.params["const"])  # percent per day

    alpha_ann_pct = (
        (1.0 + alpha_daily_pct / RET_SCALE) ** OBS_YEAR - 1.0
    ) * RET_SCALE

    out = {
        "symbol":  asset,
        "nobs":    len(df_ret),
        "corr":    float(y.corr(x)),
        "beta":    beta,
        "alpha":   alpha_ann_pct,
        "ann_ret": ann_ret,
    }
    out.update(series_stats(y))
    return out

# -------------------------------------------------------------------
# compute statistics table
# -------------------------------------------------------------------
rows = [compute_stats(tkr, benchmark, df_prices) for tkr in tickers]
df_stats = pd.DataFrame(rows).set_index("symbol")[[
    "nobs", "ann_ret", "vol", "skew", "kurt",
    "min", "max", "corr", "beta", "alpha"
]]

pd.set_option("display.width", 160)
print("\n" + df_stats.to_string())
