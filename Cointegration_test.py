# pip install yfinance pandas numpy statsmodels matplotlib
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# ---------- Data ----------
def download_prices(tickers, start="2015-01-01", end=None, interval="1d"):
    """
    Returns a DataFrame of Adjusted Close prices indexed by date.
    """
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    # yfinance returns multi-index columns when multiple tickers; handle both cases
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Adj Close"].copy()
    else:
        prices = df.rename(columns={"Adj Close": tickers[0]})["Adj Close"].to_frame()
    # Drop columns with too many NaNs, forward/back fill small gaps
    prices = prices.dropna(axis=1, how="all")
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    # Keep only dates with data for all retained tickers
    prices = prices.dropna(axis=0, how="any")
    return prices

# ---------- Unit root helpers (ADF) ----------
def _adf_pvalue(series, autolag="AIC", maxlag=None, regression="c"):
    """
    Return ADF p-value for a series (dropna). Default includes a constant (regression='c').
    """
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return np.nan
    try:
        stat, pval, *_ = adfuller(s, autolag=autolag, maxlag=maxlag, regression=regression)
        return float(pval)
    except Exception:
        return np.nan

def classify_integration(series, alpha=0.05):
    """
    Classify as I(0), I(1), or 'unclear' using ADF:
      - If levels reject unit root (p<alpha) => I(0)
      - Else if first-diff rejects unit root (p<alpha) => I(1)
      - Else => 'unclear'
    Returns (label, p_level, p_diff)
    """
    p_level = _adf_pvalue(series, regression="c")
    p_diff  = _adf_pvalue(pd.Series(series).diff(), regression="c")
    if not np.isnan(p_level) and p_level < alpha:
        label = "I(0)"
    elif (np.isnan(p_level) or p_level >= alpha) and (not np.isnan(p_diff) and p_diff < alpha):
        label = "I(1)"
    else:
        label = "unclear"
    return label, p_level, p_diff

def pretest_integration(prices, alpha=0.05):
    """
    Run ADF-based classification for every column in prices.
    Returns a DataFrame with columns: ['ticker','integration','ADF_level_p','ADF_diff_p']
    """
    rows = []
    for t in prices.columns:
        label, pL, pD = classify_integration(prices[t], alpha=alpha)
        rows.append({"ticker": t, "integration": label, "ADF_level_p": pL, "ADF_diff_p": pD})
    return pd.DataFrame(rows).set_index("ticker")

# ---------- Stats helpers ----------
def ols_hedge_ratio(y, x, add_const=True):
    X = sm.add_constant(x) if add_const else x
    model = sm.OLS(y, X).fit()
    if add_const:
        const, beta = model.params.iloc[0], model.params.iloc[1]
    else:
        const, beta = 0.0, model.params.iloc[0]
    return const, beta, model

def spread_from_beta(y, x, const, beta):
    return y - (const + beta * x)

def half_life(spread):
    """
    Estimate half-life of mean reversion using AR(1) on spread differences.
    Δs_t = α + ρ*s_{t-1} + ε; half-life = -ln(2)/ln(1+ρ)
    """
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return np.nan
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna().loc[ds.index]
    X = sm.add_constant(s_lag)
    rho = sm.OLS(ds, X).fit().params.get(s_lag.name, np.nan)
    if rho is None or np.isnan(rho) or (1 + rho) <= 0:
        return np.nan
    return -np.log(2) / np.log(1 + rho)

def zscore(series):
    s = pd.Series(series)
    return (s - s.mean()) / s.std(ddof=0)

# ---------- Cointegration test with integration pre-check ----------
def eg_cointegration_results(
    prices,
    max_pairs=None,
    add_const=True,
    adf_on_spread=True,
    alpha=0.05,
    require_I1=True,           # only test pairs where both series look I(1)
    include_stationary_pairs=True  # report I(0)+I(0) linear relations
):
    """
    Test all pairs (or first max_pairs) for Engle–Granger cointegration.
    Adds ADF-based integration classification per ticker and, optionally, filters to I(1)+I(1).
    """
    integ = pretest_integration(prices, alpha=alpha)

    tickers = list(prices.columns)
    pairs = list(itertools.combinations(tickers, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    rows = []
    for a, b in pairs:
        ya = prices[a]; xb = prices[b]
        a_label = integ.loc[a, "integration"]
        b_label = integ.loc[b, "integration"]

        if require_I1 and not (a_label == "I(1)" and b_label == "I(1)"):
            if include_stationary_pairs and (a_label == "I(0)" and b_label == "I(0)"):
                const_s, beta_s, model_s = ols_hedge_ratio(ya, xb, add_const=add_const)
                spr_s = spread_from_beta(ya, xb, const_s, beta_s)
                hl_s = half_life(spr_s)
                adf_s_stat, adf_s_p = (np.nan, np.nan)
                if adf_on_spread:
                    try:
                        adf_s_stat, adf_s_p, *_ = adfuller(spr_s.dropna(), autolag="AIC")
                    except Exception:
                        pass
                rows.append({
                    "pair_type": "stationary_pair",
                    "y": a, "x": b,
                    "y_integration": a_label, "x_integration": b_label,
                    "y_ADF_level_p": integ.loc[a, "ADF_level_p"],
                    "y_ADF_diff_p":  integ.loc[a, "ADF_diff_p"],
                    "x_ADF_level_p": integ.loc[b, "ADF_level_p"],
                    "x_ADF_diff_p":  integ.loc[b, "ADF_diff_p"],
                    "beta(y~x)": beta_s,
                    "const": const_s,
                    "EG_stat": np.nan, "EG_pvalue": np.nan,
                    "ADF_spread_stat": adf_s_stat, "ADF_spread_pvalue": adf_s_p,
                    "spread_vol": float(np.std(spr_s.dropna(), ddof=1)),
                    "half_life_days": hl_s,
                    "R2_y_on_x": model_s.rsquared,
                    "n_obs": int(spr_s.dropna().shape[0]),
                })
            continue

        # Direction 1: y=a on x=b
        const, beta, model = ols_hedge_ratio(ya, xb, add_const=add_const)
        spr = spread_from_beta(ya, xb, const, beta)
        EG_stat, EG_p, _ = coint(ya, xb)
        adf_stat, adf_p = (np.nan, np.nan)
        if adf_on_spread:
            try:
                adf_stat, adf_p, *_ = adfuller(spr.dropna(), autolag="AIC")
            except Exception:
                pass
        hl = half_life(spr)
        rows.append({
            "pair_type": "I(1)+I(1)_tested",
            "y": a, "x": b,
            "y_integration": a_label, "x_integration": b_label,
            "y_ADF_level_p": integ.loc[a, "ADF_level_p"],
            "y_ADF_diff_p":  integ.loc[a, "ADF_diff_p"],
            "x_ADF_level_p": integ.loc[b, "ADF_level_p"],
            "x_ADF_diff_p":  integ.loc[b, "ADF_diff_p"],
            "beta(y~x)": beta,
            "const": const,
            "EG_stat": EG_stat, "EG_pvalue": EG_p,
            "ADF_spread_stat": adf_stat, "ADF_spread_pvalue": adf_p,
            "spread_vol": float(np.std(spr.dropna(), ddof=1)),
            "half_life_days": hl,
            "R2_y_on_x": model.rsquared,
            "n_obs": int(spr.dropna().shape[0]),
        })

        # Direction 2: y=b on x=a
        const2, beta2, model2 = ols_hedge_ratio(xb, ya, add_const=add_const)
        spr2 = spread_from_beta(xb, ya, const2, beta2)
        EG_stat2, EG_p2, _ = coint(xb, ya)
        adf_stat2, adf_p2 = (np.nan, np.nan)
        if adf_on_spread:
            try:
                adf_stat2, adf_p2, *_ = adfuller(spr2.dropna(), autolag="AIC")
            except Exception:
                pass
        hl2 = half_life(spr2)
        rows.append({
            "pair_type": "I(1)+I(1)_tested",
            "y": b, "x": a,
            "y_integration": b_label, "x_integration": a_label,
            "y_ADF_level_p": integ.loc[b, "ADF_level_p"],
            "y_ADF_diff_p":  integ.loc[b, "ADF_diff_p"],
            "x_ADF_level_p": integ.loc[a, "ADF_level_p"],
            "x_ADF_diff_p":  integ.loc[a, "ADF_diff_p"],
            "beta(y~x)": beta2,
            "const": const2,
            "EG_stat": EG_stat2, "EG_pvalue": EG_p2,
            "ADF_spread_stat": adf_stat2, "ADF_spread_pvalue": adf_p2,
            "spread_vol": float(np.std(spr2.dropna(), ddof=1)),
            "half_life_days": hl2,
            "R2_y_on_x": model2.rsquared,
            "n_obs": int(spr2.dropna().shape[0]),
        })

    res = pd.DataFrame(rows)

    # Sorting logic
    if not res.empty:
        res["sort_bucket"] = np.where(res["pair_type"] == "I(1)+I(1)_tested", 0, 1)
        res = res.sort_values(
            by=["sort_bucket", "EG_pvalue", "half_life_days"],
            ascending=[True, True, True],
            na_position="last"
        ).drop(columns=["sort_bucket"]).reset_index(drop=True)
    return res

# ---------- Convenience: run everything ----------
if __name__ == "__main__":
    # Example universe — replace with your own
    TICKERS = ["XOM", "CVX", "BP", "SHEL", "TOT", "VLO", "PSX", "HES"]
    #TICKERS = ["USDNZD=X", "AUD=X"]
    START = "2021-01-01"
    END = "2023-01-01"      # or "2025-09-01"
    INTERVAL = "1d" # could also use "1wk" / "1mo" if you like
    ALPHA = 0.05

    prices = download_prices(TICKERS, start=START, end=END, interval=INTERVAL)

    results = eg_cointegration_results(
        prices,
        max_pairs=None,
        add_const=True,
        adf_on_spread=True,
        alpha=ALPHA,
        require_I1=True,                # only run EG when both look I(1)
        include_stationary_pairs=True   # also report I(0)+I(0) stationary relations
    )

    # Show top 20 candidates
    print(results.head(20).to_string(index=False))

    # === NEW: one-line verdict on cointegration at 5% ===
    if results.empty:
        print("\nCointegration @ 5%: NO RESULTS")
    else:
        tested = results[results["pair_type"] == "I(1)+I(1)_tested"]
        if tested.empty:
            print("\nCointegration @ 5%: NOT TESTED (pretests suggest at least one series is not I(1); set require_I1=False to force EG).")
        else:
            best_idx = tested["EG_pvalue"].idxmin()
            best = tested.loc[best_idx]
            verdict = "YES" if (best["EG_pvalue"] < ALPHA) else "NO"
            print(f"\nCointegration @ 5%: {verdict} (best EG p-value = {best['EG_pvalue']:.4g} for {best['y']} ~ {best['x']})")

    # Optional: pick the top row and preview spread z-score for trading bands
    if not results.empty and not np.isnan(results.iloc[0]["beta(y~x)"]):
        top = results.iloc[0]
        y = prices[top["y"]]
        x = prices[top["x"]]
        spr = y - (top["const"] + top["beta(y~x)"] * x)
        z = zscore(spr)
        print("\nTop pair:", top["y"], "/", top["x"], "| type:", top["pair_type"])
        print("Hedge ratio beta(y~x):", round(top["beta(y~x)"], 4), " | const:", round(top["const"], 4))
        print("Half-life (days):", round(top["half_life_days"], 2))
        print("Latest spread z-score:", round(float(z.iloc[-1]), 2))

