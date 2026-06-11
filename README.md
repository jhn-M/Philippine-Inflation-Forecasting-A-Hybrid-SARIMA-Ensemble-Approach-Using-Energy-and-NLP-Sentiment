# Macro Nexus Forecasting
### Forecasting Philippine Inflation Using Oil Prices and NLP Sentiment Scores
**Hybrid SARIMA + Machine Learning Model**

---

## Problem Statement

Inflation forecasting is critical for banks, financial institutions, and market participants making forward-looking decisions, yet most univariate models fail to capture the external shocks that drive sudden price movements. This project builds a hybrid SARIMA model corrected by machine learning using macroeconomic indicators — Brent crude oil prices and NLP sentiment scores derived from financial news headlines — to forecast Philippine inflation one to several months ahead.

A key limitation of this approach is that while sentiment and oil prices exhibit high daily volatility that likely influences inflation, BSP reports inflation monthly, forcing all features to be aggregated at the monthly level and potentially losing granular signal in the process.

---

## Architecture Overview

### Step 1: Data Collection
- Scrape financial headlines from Business World Online using BeautifulSoup with ThreadPoolExecutor for parallel execution
- Pull inflation rates from BSP; Brent crude oil prices via yfinance

### Step 2: Data Cleaning
- Remove unnecessary symbols and stop words from headlines
- Drop NaNs for exogenous variables
- Apply rolling means and lags to prevent data leakage before merging

### Step 3: Modeling
- Fit a SARIMA baseline model using auto_arima with exhaustive search (AIC criterion)
- Train Random Forest and XGBoost correction models on SARIMA residuals to capture non-linear patterns the SARIMA cannot model
- Add each correction model's output to the SARIMA forecast to produce the final hybrid predictions
- Ensemble the two correction models by averaging their residual predictions

### Step 4: Evaluation
- Evaluate SARIMA baseline, SARIMA-RF, SARIMA-XGB, and SARIMA-XGBRF ensemble using MSE, RMSE, and MAE on a held-out test set (20% split)
- Select the best-performing model to refit on the full dataset for final forecasting

### Step 5: Final Forecast and Conclusion
- Refit SARIMA and correction models on the full dataset using the same orders
- Generate a 2-month ahead forecast using real observed exogenous values
- Document findings, limitations, and model behavior

---

## Data Sources

| Feature | Source | Date Range |
|---|---|---|
| Inflation Rate | BSP — [tab34_inf_2018](https://www.bsp.gov.ph/Statistics/Prices/tab34_inf_2018.aspx) | Jan 2021 – Mar 2026 |
| Brent Crude Oil | yfinance ticker `BZ=F` | Jan 2021 – Mar 2026 |
| Sentiment Scores | finBERT on scraped [Business World Online](https://www.bworldonline.com/banking-finance/) headlines | Jan 2021 – Mar 2026 |

> **Note on USD-PHP:** USD-PHP exchange rate was initially hypothesized as a third feature. Correlation analysis using BSP and yfinance data produced weak results (r < 0.25 across all lags), likely due to BSP monetary intervention suppressing direct pass-through effects. The feature was dropped to maintain data integrity.

---

## Feature Engineering

Both exogenous features use a **3-month rolling mean shifted by lag 2** to capture delayed macro pass-through effects while preventing data leakage.

| Feature | Transformation | Correlation with Inflation |
|---|---|---|
| Brent Crude Oil | Rolling mean (window=3), lag 2 | r = 0.67 |
| finBERT Sentiment | Rolling mean (window=3), lag 2 | r = 0.44 |

---

## SARIMA Configuration

Stationarity and order selection:

- **ADF Test:** p = 0.133 → non-stationary, d=1 required
- **Seasonal differencing (nsdiffs/OCSB):** D=0 — seasonal differencing not needed
- **ACF/PACF and seasonal decomposition:** weak but present seasonality — seasonal component retained
- **Model selected by auto_arima (exhaustive, AIC):** `SARIMAX(0, 1, 3)x(1, 0, [], 12)`

---

## Results

### Model Evaluation on Test Set

| Model | MSE | RMSE | MAE |
|---|---|---|---|
| SARIMA Baseline | 0.8155 | 0.9031 | 0.5647 |
| SARIMA-RF | 0.6797 | 0.8244 | 0.5368 |
| SARIMA-XGB | 0.6567 | 0.8104 | 0.6063 |
| **SARIMA-XGBRF (Ensemble)** | **0.6471** | **0.8044** | **0.5527** |

The SARIMA-XGBRF ensemble achieved the best performance across MSE and RMSE, representing a **10.9% reduction in RMSE** over the SARIMA baseline (0.8044 vs 0.9031). All three correction models consistently outperformed the baseline, confirming that exogenous macroeconomic features add meaningful predictive signal beyond what SARIMA captures alone.

### Baseline SARIMA Forecast
<img width="2085" height="735" alt="prelimenary_plot" src="https://github.com/user-attachments/assets/eff86a98-2474-463d-934a-3b2b93bad2ea" />


### Model Evaluation Plot
<img width="2065" height="1185" alt="prelimenary_plot1" src="https://github.com/user-attachments/assets/6899df20-8687-4ab8-884b-982b5d25761b" />

### Final 2-Month Ahead Forecast

Refit on the full dataset using real observed exogenous values for April and May 2026:

| Month | SARIMA | SARIMA-XGBRF Ensemble |
|---|---|---|
| April 2026 | 4.63% | 5.37% |
| May 2026 | 4.68% | 4.52% |

The ensemble projects a short-term spike in April followed by partial moderation in May. Both figures remain elevated relative to the low inflation environment observed in mid-2025.

---

## Conclusions and Findings

- Correlation analysis confirmed that Brent crude oil (lag 2, r=0.67) and finBERT sentiment scores (lag 2, r=0.44) carry a meaningful positive leading relationship with Philippine inflation.
- USD-PHP exchange rate was dropped after failing to show predictive correlation (r < 0.25 across all lags), documented as a data integrity decision.
- The hybrid SARIMA-XGBRF ensemble outperformed the standalone SARIMA baseline with RMSE 0.8044 vs 0.9031 — a ~10.9% improvement.
- All three correction models beat the baseline, validating the residual correction approach.

---

## Limitations

- **Monthly aggregation loss:** Oil prices and sentiment exhibit high daily volatility that is lost when averaged to monthly frequency, weakening the signal available to correction models.
- **Near-white-noise residuals:** SARIMA residuals on a well-specified model behave close to white noise, leaving minimal learnable structure for the ensemble correction layer.
- **Bounded forecast horizon:** The 2-month ahead forecast is constrained by the availability of real observed exogenous data. Forecasting beyond this window requires separate models for oil prices and sentiment.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis & Engineering | Pandas, NumPy, EDA, Feature Engineering, Data Cleaning |
| Machine Learning | Scikit-Learn, Random Forest Regressor, XGBoost, Model Evaluation |
| Time Series & Econometrics | ARIMA, SARIMA, Forecasting, Residual Modeling, ADF Stationarity Testing, Seasonal Decomposition, Statsmodels, pmdarima |
| NLP | finBERT, Sentiment Analysis, Text Processing, BeautifulSoup Web Scraping |
| Visualization | Matplotlib |
