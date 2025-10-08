#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Pairs Trading Backtest (Educational Version) with Plots

Adds:
- Line chart of capital over time
- Line chart of drawdown over time
- Line chart of the SPREAD over time (with rolling mean and zero line)
- Basic performance stats (CAGR, Sharpe, Max Drawdown, Win rate)
- Saves charts to PNG files

Core design remains intentionally simple for clarity:
- Full-sample alpha/beta (look-ahead for simplicity)
- Rolling z-score signal
- One-bar execution lag
- No transaction costs
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------------------
# User settings
# -------------------------------
TICKERS = ["XOM", "VLO"]   # [Y, X] order
START   = "2023-01-01"
END     = "2025-01-01"                    # or "2025-09-01"
INTERVAL = "1d"
Z_WINDOW = 60                     # rolling window for mean/std of spread (days)
Z_ENTRY  = 2.0                    # entry threshold
Z_EXIT   = 0.5                    # exit threshold
GROSS_EXPOSURE = 1.0              # target gross exposure when in a position
INITIAL_CAPITAL = 100.0           # starting capital ($)

# -------------------------------
# Helpers
# -------------------------------
def download_prices(tickers, start="2015-01-01", end=None, interval="1d"):
    """Return DataFrame of Adjusted Close prices indexed by date."""
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Adj Close"].copy()
    else:
        # Single ticker path
        prices = df.rename(columns={"Adj Close": tickers[0]})["Adj Close"].to_frame()
    return prices

def ols_hedge_ratio(y, x, add_const=True):
    """Return (alpha, beta, fitted_model) from OLS of y on x."""
    X = sm.add_constant(x) if add_const else x
    model = sm.OLS(y, X).fit()
    if add_const:
        const, beta = model.params.iloc[0], model.params.iloc[1]
    else:
        const, beta = 0.0, model.params.iloc[0]
    return const, beta, model

def rolling_zscore(series, window):
    """Rolling z-score: (x - rolling_mean) / rolling_std with ddof=0."""
    s = series.astype(float)
    m = s.rolling(window=window, min_periods=window).mean()
    v = s.rolling(window=window, min_periods=window).std(ddof=0)
    z = (s - m) / v
    return z

def scale_weights(raw_wY, raw_wX, gross_target=1.0):
    """Scale (wY, wX) so that |wY|+|wX| == gross_target (preserve ratio)."""
    denom = abs(raw_wY) + abs(raw_wX)
    if denom == 0:
        return 0.0, 0.0
    return gross_target * raw_wY / denom, gross_target * raw_wX / denom

def compute_drawdown(cum_returns):
    """Compute drawdown series from cumulative wealth (capital)."""
    running_max = cum_returns.cummax()
    dd = (cum_returns / running_max) - 1.0
    return dd

# -------------------------------
# Backtest
# -------------------------------
def backtest_simple_pairs(prices, y_ticker, x_ticker,
                          z_window=60, z_entry=2.0, z_exit=0.5,
                          gross_exposure=1.0, initial_capital=100.0):
    """
    Very simple backtest with:
    - Static alpha/beta on full sample (IN-SAMPLE).
    - Rolling z-score for signals.
    - One-bar execution lag (signal at t, trade at t+1).
    - No costs/slippage/financing.
    """
    y = prices[y_ticker].astype(float)
    x = prices[x_ticker].astype(float)

    # 1) Static hedge ratio (full-sample) -- simple but look-ahead in practice
    alpha, beta, model = ols_hedge_ratio(y, x, add_const=True)

    # 2) Spread and rolling z-score
    spread = y - (alpha + beta * x)
    z = rolling_zscore(spread, window=z_window)

    # 3) Signals at time t
    #    +1 = long spread (buy Y, sell beta*X), -1 = short spread, 0 = flat
    signal = pd.Series(0, index=prices.index, dtype=float)
    signal[z >  z_entry]  = -1.0
    signal[z < -z_entry]  = +1.0
    signal[abs(z) < z_exit] = 0.0

    # 4) Convert signals to positions with 1-bar execution lag
    raw_wY = pd.Series(0.0, index=signal.index)
    raw_wX = pd.Series(0.0, index=signal.index)
    raw_wY[signal ==  1.0] = +1.0
    raw_wX[signal ==  1.0] = -beta
    raw_wY[signal == -1.0] = -1.0
    raw_wX[signal == -1.0] = +beta

    # Execution lag: trade tomorrow -> shift weights by 1 day forward
    raw_wY = raw_wY.shift(1).fillna(0.0)
    raw_wX = raw_wX.shift(1).fillna(0.0)

    # 5) Scale to target gross exposure
    wY = pd.Series(0.0, index=raw_wY.index)
    wX = pd.Series(0.0, index=raw_wX.index)
    for t in raw_wY.index:
        wY[t], wX[t] = scale_weights(raw_wY[t], raw_wX[t], gross_target=gross_exposure)

    # 6) Compute daily simple returns of Y and X
    rY = y.pct_change().fillna(0.0)
    rX = x.pct_change().fillna(0.0)

    # Strategy return (no costs): r = wY * rY + wX * rX
    strat_r = (wY * rY) + (wX * rX)

    # 7) Capital path
    cap = (1.0 + strat_r).cumprod() * initial_capital
    cap.iloc[0] = initial_capital  # ensure exact start

    out = pd.DataFrame({
        "Y": y,
        "X": x,
        "alpha": alpha,
        "beta": beta,
        "spread": spread,
        "zscore": z,
        "signal_t": signal,           # signal evaluated at t (pre-lag)
        "wY": wY,
        "wX": wX,
        "rY": rY,
        "rX": rX,
        "strat_return": strat_r,
        "capital": cap
    })

    return out, alpha, beta

def summarize_and_plot(bt_df, start_cap=100.0, save_prefix="pairs", z_window=60):
    """Print simple stats and save plots (capital, drawdown, spread)."""
    # Basic stats
    r = bt_df["strat_return"].copy()
    cap = bt_df["capital"].copy()
    dd = compute_drawdown(cap)

    n = r.shape[0]
    if n <= 1:
        print("Not enough data to summarize.")
        return

    # Sharpe (daily -> annualized, assuming 252 trading days)
    mu = r.mean()
    sd = r.std(ddof=0)
    ann_mu = mu * 252
    ann_sd = sd * np.sqrt(252)
    sharpe = ann_mu / ann_sd if ann_sd > 0 else np.nan

    # CAGR
    years = n / 252.0
    cagr = (cap.iloc[-1] / start_cap)**(1/years) - 1 if years > 0 else np.nan

    # Max Drawdown
    max_dd = dd.min()

    # Win rate (by day)
    win_rate = (r > 0).mean()

    print("\n=== Performance Summary ===")
    print(f"Days: {n}")
    print(f"Final capital: ${cap.iloc[-1]:,.2f}")
    print(f"CAGR: {100*cagr:.2f}%")
    print(f"Sharpe (ann.): {sharpe:.2f}")
    print(f"Max Drawdown: {100*max_dd:.2f}%")
    print(f"Win rate (daily): {100*win_rate:.2f}%")

    # --- Plot capital ---
    plt.figure()
    cap.plot(title="Capital Over Time")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.tight_layout()
    cap_path = f"{save_prefix}_capital.png"
    plt.savefig(cap_path)
    print(f"Saved capital chart to {cap_path}")

    # --- Plot drawdown ---
    plt.figure()
    dd.plot(title="Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    dd_path = f"{save_prefix}_drawdown.png"
    plt.savefig(dd_path)
    print(f"Saved drawdown chart to {dd_path}")

    # --- Plot spread (raw) with rolling mean and zero line ---
    plt.figure()
    spread = bt_df["spread"].copy()
    spread.plot(title="Spread Over Time")
    # rolling mean with same window used for z-score
    spread.rolling(window=z_window, min_periods=z_window).mean().plot()
    # zero line
    ax = plt.gca()
    ax.axhline(0.0)
    plt.xlabel("Date")
    plt.ylabel("Spread (Y - (alpha + beta*X))")
    plt.legend(["Spread", f"Rolling Mean ({z_window})", "Zero"])
    plt.tight_layout()
    sp_path = f"{save_prefix}_spread.png"
    plt.savefig(sp_path)
    print(f"Saved spread chart to {sp_path}")

# -------------------------------
# Run (if invoked directly)
# -------------------------------
if __name__ == "__main__":
    prices = download_prices(TICKERS, start=START, end=END, interval=INTERVAL)

    # Ensure we have exactly two columns and name them (Y first, then X)
    if len(TICKERS) != 2:
        raise ValueError("Please provide exactly two tickers in TICKERS for this simple demo.")

    y_ticker, x_ticker = TICKERS[0], TICKERS[1]

    bt, alpha, beta = backtest_simple_pairs(
        prices,
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        z_window=Z_WINDOW,
        z_entry=Z_ENTRY,
        z_exit=Z_EXIT,
        gross_exposure=GROSS_EXPOSURE,
        initial_capital=INITIAL_CAPITAL
    )

    # --- Print summary ---
    start_date = bt.index.min().date()
    end_date   = bt.index.max().date()
    n_days     = len(bt)
    cap0 = bt["capital"].iloc[0]
    capT = bt["capital"].iloc[-1]

    print("=== Simple Pairs Backtest (Educational) ===")
    print(f"Y ~ X : {TICKERS[0]} ~ {TICKERS[1]}")
    print(f"Sample: {start_date} -> {end_date}  ({n_days} trading days)")
    print(f"Alpha (full-sample): {alpha:.6f}   Beta (full-sample): {beta:.6f}")
    print(f"Z-window: {Z_WINDOW}   Entry: ±{Z_ENTRY}   Exit: {Z_EXIT}")
    print(f"Initial capital: ${cap0:,.2f}   Final capital: ${capT:,.2f}")

    # Checkpoints at ~2,3,5 years if available (assume ~252 trading days per year)
    for years, bars in [(2, 2*252), (3, 3*252), (5, 5*252)]:
        if n_days > bars:
            capY = bt["capital"].iloc[bars]
            print(f"Capital after ~{years} years: ${capY:,.2f}")

    # Optional: save results
    out_csv = "simple_pairs_backtest_results.csv"
    bt.to_csv(out_csv, index=True)
    print(f"\nSaved full daily results to {out_csv}")

    # Summaries and plots
    summarize_and_plot(bt, start_cap=INITIAL_CAPITAL, save_prefix="pairs_demo", z_window=Z_WINDOW)
