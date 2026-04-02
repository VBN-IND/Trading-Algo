"""
Risk analysis module.

Computes standard quantitative risk metrics at the single-asset level:
  - Annualised volatility
  - Value at Risk (historical & parametric)
  - Conditional VaR (Expected Shortfall)
  - Sharpe Ratio
  - Maximum Drawdown
  - Rolling Beta (vs. benchmark)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class RiskReport:
    ticker: str
    period: str
    ann_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float          # 1-day, 95% confidence
    cvar_95: float         # Expected shortfall at 95%
    var_99: float          # 1-day, 99% confidence
    skewness: float
    kurtosis: float
    n_observations: int
    generated_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def summary(self) -> str:
        lines = [
            f"\n{'─' * 40}",
            f"  Risk Report: {self.ticker}  ({self.period})",
            f"{'─' * 40}",
            f"  Observations          : {self.n_observations} trading days",
            f"  Annualised Volatility : {self.ann_volatility:.2%}",
            f"  Sharpe Ratio          : {self.sharpe_ratio:.2f}",
            f"  Max Drawdown          : {self.max_drawdown:.2%}",
            f"  VaR  (95%, 1-day)     : {self.var_95:.2%}",
            f"  CVaR (95%, 1-day)     : {self.cvar_95:.2%}",
            f"  VaR  (99%, 1-day)     : {self.var_99:.2%}",
            f"  Return Skewness       : {self.skewness:.3f}",
            f"  Excess Kurtosis       : {self.kurtosis:.3f}",
            f"{'─' * 40}",
        ]
        return "\n".join(lines)


class RiskAnalyser:
    """
    Computes risk metrics from a returns series.
    All methods operate on daily simple returns (pct_change).
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 5% — approximate UK base rate).
        """
        self.rf = risk_free_rate

    def analyse(self, df: pd.DataFrame, ticker: str = "", period: str = "") -> RiskReport:
        """Run the full risk pipeline and return a RiskReport."""
        returns = df["returns"].dropna()

        return RiskReport(
            ticker=ticker or df["ticker"].iloc[0] if "ticker" in df.columns else "UNKNOWN",
            period=period,
            ann_volatility=self.annualised_volatility(returns),
            sharpe_ratio=self.sharpe_ratio(returns),
            max_drawdown=self.max_drawdown(df["close"]),
            var_95=self.var_historical(returns, confidence=0.95),
            cvar_95=self.cvar(returns, confidence=0.95),
            var_99=self.var_historical(returns, confidence=0.99),
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurtosis()),
            n_observations=len(returns),
        )

    # ── Core metrics ────────────────────────────────────────────────────────

    def annualised_volatility(self, returns: pd.Series) -> float:
        """Daily vol scaled to annual using sqrt(252)."""
        return float(returns.std() * np.sqrt(TRADING_DAYS))

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualised Sharpe: (mean_daily_return - daily_rf) / daily_vol * sqrt(252)."""
        daily_rf = self.rf / TRADING_DAYS
        excess = returns.mean() - daily_rf
        vol = returns.std()
        if vol == 0:
            return 0.0
        return float((excess / vol) * np.sqrt(TRADING_DAYS))

    def max_drawdown(self, prices: pd.Series) -> float:
        """Maximum peak-to-trough percentage decline."""
        roll_max = prices.cummax()
        drawdown = (prices - roll_max) / roll_max
        return float(drawdown.min())

    def var_historical(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical (non-parametric) VaR — percentile of observed losses."""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def var_parametric(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Parametric VaR — assumes normally distributed returns."""
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(1 - confidence)
        return float(mu + z * sigma)

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall) — mean of returns below VaR threshold.
        More conservative and coherent than VaR alone.
        """
        var = self.var_historical(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if not tail.empty else var

    # ── Rolling metrics ──────────────────────────────────────────────────────

    def rolling_volatility(self, returns: pd.Series, window: int = 21) -> pd.Series:
        """Rolling N-day annualised volatility."""
        return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    def rolling_sharpe(self, returns: pd.Series, window: int = 63) -> pd.Series:
        """Rolling N-day Sharpe ratio."""
        daily_rf = self.rf / TRADING_DAYS
        roll_mean = returns.rolling(window).mean() - daily_rf
        roll_std = returns.rolling(window).std()
        return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)

    def rolling_beta(self, returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
        """
        Rolling Beta vs. a benchmark (e.g. SPY).
        Beta = Cov(asset, benchmark) / Var(benchmark)
        """
        cov = returns.rolling(window).cov(benchmark_returns)
        var = benchmark_returns.rolling(window).var()
        return cov / var

    def drawdown_series(self, prices: pd.Series) -> pd.Series:
        """Full drawdown time series (not just the max)."""
        roll_max = prices.cummax()
        return (prices - roll_max) / roll_max
