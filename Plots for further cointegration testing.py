# pip install yfinance pandas numpy statsmodels matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Download example data (replace with your file or tickers) ---
TICKERS = ["VLO", "XOM"]
raw = yf.download(TICKERS, start="2021-01-01", end="2023-01-01", auto_adjust=False, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    data = raw["Adj Close"].copy()
else:
    data = raw[["Adj Close"]].rename(columns={"Adj Close": TICKERS[0]})

data = data.dropna()

y = data[TICKERS[0]]
x = data[TICKERS[1]]

# --- Full-sample regression to get spread ---
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()
alpha, beta = res.params
spread = y - (alpha + beta * x)

# --- Unit root tests ---
def adf_test(series, name="Series"):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF Test for {name}: stat={result[0]:.3f}, p={result[1]:.4f}")

def kpss_test(series, name="Series"):
    result = kpss(series.dropna(), regression="c")
    print(f"KPSS Test for {name}: stat={result[0]:.3f}, p={result[1]:.4f}")

print("=== UNIT ROOT TESTS ===")
adf_test(y, "Y")
adf_test(x, "X")
adf_test(spread, "spread")
kpss_test(spread, "spread")

# --- Johansen cointegration test ---
print("\n=== JOHANSEN TEST ===")
jres = coint_johansen(data[TICKERS], det_order=0, k_ar_diff=1)
print("Eigenvalues:", jres.eig)
print("Trace statistics:", jres.lr1)
print("5% crit values:", jres.cvt[:,1])

# --- Rolling ADF p-values on spread ---
window = 252  # 1-year window
adf_pvals = []
for i in range(window, len(spread)):
    sub = spread.iloc[i-window:i]
    try:
        adf_pvals.append(adfuller(sub.dropna(), autolag="AIC")[1])
    except:
        adf_pvals.append(np.nan)
adf_pvals = pd.Series(adf_pvals, index=spread.index[window:])

# --- Plots ---
plt.figure(figsize=(10,4))
spread.plot(title="spread Over Time")
plt.axhline(0, color="k", linestyle="--")
plt.show()

plt.figure(figsize=(10,4))
adf_pvals.plot(title="Rolling ADF p-values (window=252)")
plt.axhline(0.05, color="r", linestyle="--", label="5% threshold")
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plot_acf(spread.dropna(), lags=40)
plt.show()

plt.figure(figsize=(10,4))
plot_pacf(spread.dropna(), lags=40)
plt.show()
