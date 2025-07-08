"""
xbeta.py  --  beta, alpha, annualized volatility, annualized return,
              correlation, skewness, kurtosis, min / max daily return,
              and observation count for tickers versus a benchmark.
              ALSO saves a DataFrame of closing prices to *out_prices_file*.

ASCII only.  Needs yfinance, numpy, pandas, statsmodels.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm

pd.options.display.float_format = "{:.3f}".format

# -------------------------------------------------------------------
# constants
# -------------------------------------------------------------------
OBS_YEAR  = 252.0                    # trading days in one year
RET_SCALE = 100.0                    # percent-scaled returns
VOL_ANN   = np.sqrt(OBS_YEAR)
START     = "2010-01-01"             # start date
END       = pd.Timestamp.today().strftime("%Y-%m-%d")   # today
OUT_PRICES_FILE = "out_prices.csv"   # file to save closing prices

print("start :", START)
print("end   :", END)

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def get_price_series(ticker, start=START, end=END, auto_adjust=True):
    """Download daily prices for *ticker* and return Series."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=auto_adjust,
        group_by="column",
    )
    px = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    px.name = ticker
    return px

def annualized_return(prices):
    """Geometric annualized return (percent)."""
    if len(prices) < 2:
        return np.nan
    gross = prices.pct_change().dropna() + 1.0
    years = len(gross) / OBS_YEAR
    cagr  = gross.prod() ** (1.0 / years) - 1.0
    return float(cagr.iloc[0] * RET_SCALE)

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


def compute_beta_stats(asset, benchmark, start=START, end=END):
    """Return full stats (incl. alpha) for *asset* vs *benchmark*."""
    s_asset = get_price_series(asset, start, end)
    ann_ret = annualized_return(s_asset)

    # -----------------------------------------------------------
    # asset equals benchmark
    # -----------------------------------------------------------
    if asset == benchmark:
        ret = RET_SCALE * s_asset.pct_change().dropna()
        stats = {
            "symbol":  asset,
            "nobs":    len(ret),
            "corr":    1.0,
            "beta":    1.0,
            "alpha":   0.0,
            "ann_ret": ann_ret,
        }
        stats.update(series_stats(ret))
        return stats

    # -----------------------------------------------------------
    # asset differs from benchmark
    # -----------------------------------------------------------
    s_bench = get_price_series(benchmark, start, end)

    df_px  = pd.concat([s_asset, s_bench], axis=1).dropna()
    df_ret = RET_SCALE * df_px.pct_change().dropna()

    y = df_ret[asset].squeeze()
    x = df_ret[benchmark].squeeze()

    model = sm.OLS(y, sm.add_constant(x)).fit()
    beta  = float(model.params[benchmark])
    alpha_daily_pct = float(model.params["const"])  # percent per day

    # annualize alpha using compounding
    alpha_daily_frac = alpha_daily_pct / RET_SCALE
    alpha_ann_frac   = (1.0 + alpha_daily_frac) ** OBS_YEAR - 1.0
    alpha_ann_pct    = alpha_ann_frac * RET_SCALE

    stats = {
        "symbol":  asset,
        "nobs":    len(df_ret),
        "corr":    float(y.corr(x)),
        "beta":    beta,
        "alpha":   alpha_ann_pct,
        "ann_ret": ann_ret,
    }
    stats.update(series_stats(y))
    return stats


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == "__main__":
    tickers = [
        "VOO", "SPY", "IVV", "VXX", "HYG", "LQD", "IEF",
        "CAR", "CVNA", "AEVA", "CYN", "IBIT", "MSTR"
    ]
    benchmark = "VOO"
    print("benchmark:", benchmark)

    # --------- collect closing prices and write to CSV -------------
    price_series_list = [get_price_series(tkr) for tkr in tickers]
    df_prices = pd.concat(price_series_list, axis=1).dropna(how="all")
    if OUT_PRICES_FILE is not None:
        df_prices.to_csv(OUT_PRICES_FILE, float_format="%.5f")
        print("closing prices written to", OUT_PRICES_FILE)

    # --------- compute statistics table ----------------------------
    rows = [compute_beta_stats(tkr, benchmark) for tkr in tickers]
    df_stats = pd.DataFrame(rows).set_index("symbol").loc[tickers]

    wanted_cols = [
        "nobs", "ann_ret", "vol", "skew", "kurt",
        "min", "max", "corr", "beta", "alpha"
    ]
    df_out = df_stats[wanted_cols]

    pd.set_option("display.width", 160)
    print("\n" + df_out.to_string())
