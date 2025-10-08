# 🧠 Cointegration-Based Pairs Trading Strategy

## Overview
This repository implements a **statistical arbitrage strategy** based on **cointegration analysis** between two time series (typically stock prices). 
The strategy identifies long-term equilibrium relationships, then exploits short-term deviations for potential profit.

Using **Engle-Granger two-step cointegration testing**, we model the spread between two assets as:

\[
\text{Spread} = Y_t - (\alpha + \beta X_t)
\]

where \( Y_t \) and \( X_t \) are the log prices of two correlated assets.

When the spread deviates significantly from its mean, a **mean-reversion trade** is triggered — going long the undervalued asset and short the overvalued one.

---

## 🧩 Key Components

### 1. `Cointegration_test.py`
- Loads price data and performs **Engle-Granger cointegration tests**.
- Calculates regression coefficients (\( \alpha, \beta \)).
- Tests for stationarity of the spread using **ADF tests**.
- Saves results and spread data for subsequent analysis.

### 2. `Plots for further cointegration testing.py`
- Visualizes spread dynamics, drawdowns, and rolling behavior.
- Produces diagnostic plots to verify stationarity and potential entry/exit levels.

### 3. `Trading_Impl_with_Graphs.py`
- Implements the **trading logic**:
  - Computes rolling mean and standard deviation of the spread.
  - Generates **entry/exit signals** based on z-score thresholds.
  - Simulates portfolio capital evolution.
- Includes key visualizations for performance assessment.

---

## 📊 Generated Visuals

| Chart | Description |
|-------|--------------|
| ![Spread](results/pairs_demo_spread.png) | **Spread Over Time** — shows how the spread fluctuates around its mean. |
| ![Drawdown](results/pairs_demo_drawdown.png) | **Drawdown Curve** — highlights the maximum declines from historical peaks. |
| ![Bands](results/cointegra_enhanced_spread.png) | **Spread with Entry/Exit Bands** — visualizes long and short signals. |
| ![Capital](results/pairs_demo_capital.png) | **Capital Over Time** — simulates equity curve under the trading strategy. |

---

## ⚙️ Methodology

1. **Data Preparation**
   - Import price data for two correlated assets.
   - Convert prices to log-scale for stationarity.

2. **Cointegration Analysis**
   - Regress \( Y \) on \( X \) to find hedge ratio \( \beta \).
   - Test for cointegration using the ADF test on residuals.
   - If the spread is stationary → assets are cointegrated.

3. **Trading Strategy**
   - Compute rolling mean and standard deviation of spread.
   - Define bands:
     - Upper band = mean + k·σ
     - Lower band = mean − k·σ
   - **Enter trades** when spread deviates beyond bands.
   - **Exit trades** when spread reverts toward the mean.

4. **Performance Evaluation**
   - Track cumulative returns, drawdowns, and capital evolution.
   - Visualize results to assess robustness and timing of signals.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cointegration-trading.git
   cd cointegration-trading
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run scripts in order:
   ```bash
   python scripts/Cointegration_test.py
   python scripts/Plots_for_testing.py
   python scripts/Trading_Impl_with_Graphs.py
   ```

4. Outputs (plots & data) will be saved automatically in the `results/` folder.

---

## 📈 Example Results
The backtest demonstrates periods of profitable mean reversion, with moderate drawdowns and steadily increasing capital, as shown in the included plots.

---

## 🧹 Folder Structure

```
cointegration-trading/
│
├── data/                        # (Optional) Price data files (CSV)
├── notebooks/                   # If you have Jupyter notebooks
├── scripts/
│   ├── Cointegration_test.py
│   ├── Plots_for_testing.py
│   ├── Trading_Impl_with_Graphs.py
│
├── results/
│   ├── pairs_demo_spread.png
│   ├── pairs_demo_drawdown.png
│   ├── cointegra_enhanced_spread.png
│   ├── pairs_demo_capital.png
│
├── README.md                    # Explanation of your project (this file)
├── requirements.txt             # List of Python dependencies
└── .gitignore                   # Optional, to exclude temp files or data
```

---

## 🧰 Requirements

Add these to your `requirements.txt` file:
```
pandas
numpy
matplotlib
statsmodels
```

---

## 📜 License

This project is open-source under the MIT License.

---

## 💬 Author

Created by [Your Name].  
If you find this project interesting, feel free to fork it or connect via LinkedIn!
