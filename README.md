# Cointegration-Based Pairs Trading Strategy

## Overview
This repository contains the Python code for a **statistical arbitrage trading strategy** based on **cointegration analysis** between two time series. Our strategy is based on stocks  for companies the oil industry, but further extensions are possible in FX markets, other equity sectors or individual stock prices in differerent exchanges. 

Following the work of Engle & Granger (1987) and Vidyamurthy (2004), the strategy identifies long-term equilibrium relationships between two assets, then exploits short-term deviations for profit.

Using **Engle-Granger two-step cointegration testing**, we model the spread between two assets as:

\[
\text{Spread} = Y_t - (\alpha + \beta X_t)
\] 

where \( Y_t \) and \( X_t \) are the log prices of two correlated assets.

When the spread deviates significantly from its mean, a **mean-reversion trade** is triggered — going long the undervalued asset and short the overvalued one.

---

## Files

### 1. `Cointegration_test.py`
- Loads price data of a basket of securities from yahoo finance, for the chosen time period
- Performs ADF test to identify unit roots and **Engle-Granger cointegration tests**.
- Returns the securities which are most likely be to cointegrated according to EG tests


### 2. `Plots for further cointegration testing.py`
- We further investigate the hypothesis of cointegration via additional testing
- In addition to ADF tests, we perform KPSS test to confirm non-stationarity of the series in levels
- We procude diagnostic plots to verify stationarity of the spread, together with ACF and PACF function to further confirm the absence of a unit root

### 3. `Trading_Impl_with_Graphs.py`
- This last Python file implements the **trading strategy**
  - Computes rolling mean and standard deviation of the spread.
  - Generates **entry/exit signals** based on z-score thresholds.
  - Simulates the evolution of investor's capital in an examplary portfolio.
  - Returns graphs to visualize PnL and drawdowns

---

## 📊 Generated Visuals

| Chart | Description |
|-------|--------------|
| ![Spread](results/pairs_demo_spread.png) | **Spread Over Time** — shows how the spread fluctuates around its mean. |
| ![Drawdown](results/pairs_demo_drawdown.png) | **Drawdown Curve** — highlights the maximum declines from historical peaks. |
| ![Bands](results/cointegra_enhanced_spread.png) | **Spread with Entry/Exit Bands** — visualizes long and short signals. |
| ![Capital](results/pairs_demo_capital.png) | **Capital Over Time** — simulates PnL under the trading strategy. |

---

## Methodology

1. **Data Preparation**
   - Import price data for two correlated assets.

2. **Cointegration Analysis**
   - Regress \( Y \) on \( X \) to find hedge ratio \( \beta \).
   - Test for presence of unit root in the assets' series in levels.
   - Once found evindence of unit root, test for cointegration via EG tests, testing for unit root in the regression's residual (i.e., in the spread)
   - If the spread is stationary → assets are cointegrated.

3. **Trading Strategy**
   - Compute rolling mean and standard deviation of spread.
   - Define bands:
     - Upper band = mean + k·σ
     - Lower band = mean − k·σ
   - **Enter trades** when spread deviates beyond bands.
   - **Exit trades** when spread reverts toward the mean.

4. **Monitoring Performance**
   - Track cumulative returns, drawdowns, and capital evolution.
   - Visualize results to assess robustness and timing of signals.

---

If you find this project interesting, feel free to fork it or connect via LinkedIn! https://www.linkedin.com/in/simonedeluca00/
