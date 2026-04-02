# Trading-Algo - Risk Part — Automated Market Data Ingestion & Predictive Risk Analysis.

> **Status: Active Development** — Core ingestion pipeline and risk modules operational. Backtesting engine and live signal generation in progress.

A Python-based algorithmic trading framework focused on automated market data ingestion, portfolio risk modelling, and predictive volatility analysis.

---

## Overview

AlgoRisk ingests real-time and historical market data, computes risk metrics (VaR, CVaR, Sharpe, Beta), and applies statistical models to forecast short-term volatility and drawdown risk across equity portfolios.

**Core goals:**
- Automate end-to-end data ingestion from multiple market sources
- Build a reproducible risk analysis pipeline (position-level → portfolio-level)
- Lay groundwork for signal generation based on risk-adjusted returns

---

## Architecture

```
trading-algo/
├── src/
│   ├── ingestion/       # Market data fetchers & normalisation
│   │   ├── fetcher.py   # Yahoo Finance + extensible source layer
│   │   └── pipeline.py  # Scheduling & data storage logic
│   ├── analysis/        # Risk metrics & predictive models
│   │   ├── risk.py      # VaR, CVaR, Sharpe, Beta, drawdown
│   │   └── forecast.py  # EWMA volatility + rolling z-score signals
│   └── utils/
│       ├── config.py    # Config management
│       └── logger.py    # Structured logging
├── data/                # Local cache (git-ignored)
├── tests/               # Unit tests (pytest)
├── notebooks/           # Exploratory analysis
├── main.py              # CLI entry point
└── requirements.txt
```

---

##  Features

###  Implemented
- **Market Data Ingestion** — Fetches OHLCV data for any ticker via `yfinance`; normalised into consistent pandas DataFrames
- **Rolling Risk Metrics** — Value at Risk (historical & parametric), Conditional VaR, Sharpe Ratio, max drawdown, rolling Beta
- **Volatility Forecasting** — EWMA (Exponentially Weighted Moving Average) model for forward-looking vol estimates
- **Z-Score Signals** — Flags statistically significant price deviations from rolling mean
- **CLI Interface** — Run full analysis on any ticker from the command line

### In Progress
- Multi-asset portfolio risk aggregation
- Correlation matrix & diversification scoring
- Backtesting engine with configurable strategies
- Alerting layer (email/Slack) for risk threshold breaches

---

##  Quickstart

```bash
# Cloning & installing
git clone https://github.com/VBN-IND/trading-algo.git
cd trading-algo
pip install -r requirements.txt

# Running risk analysis on a ticker
python main.py --ticker AAPL --period 1y

# Running on multiple tickers
python main.py --ticker AAPL MSFT GOOGL --period 6mo
```

**Example output:**
```
[AlgoRisk] Fetching data for AAPL (period=1y)...
[AlgoRisk] Data ingested: 252 trading days

--- Risk Report: AAPL ---
Annualised Volatility : 28.43%
Sharpe Ratio          : 1.14
Max Drawdown          : -12.67%
VaR (95%, 1-day)      : -2.31%
CVaR (95%, 1-day)     : -3.58%
Current Z-Score       : -0.82  [NEUTRAL]
EWMA Vol Forecast     : 27.91%
```

---

##  Requirements

```
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.11.0
matplotlib>=3.8.0
```

---

##  Roadmap

- [ ] Live data streaming via WebSocket
- [ ] Portfolio-level VaR with correlation adjustments
- [ ] ML volatility model (GARCH via `arch` library)
- [ ] Simple mean-reversion strategy + backtest
- [ ] Dashboard (Streamlit or FastAPI + React)

---

##  Notes

This is a personal research project built to deepen understanding of quantitative risk modelling and systematic trading infrastructure. Not financial advice.

---

*Built with Python · pandas · NumPy · SciPy · yfinance*
