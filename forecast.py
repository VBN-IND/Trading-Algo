"""
Volatility forecasting & signal generation.

Implements:
  - EWMA (Exponentially Weighted Moving Average) volatility model
  - Rolling z-score mean-reversion signal
  - Regime detection (low / normal / elevated / stress)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class ForecastResult:
    ticker: str
    ewma_vol_forecast: float       # Annualised
    zscore: float                  # Current price z-score
    signal: str                    # LONG / SHORT / NEUTRAL
    regime: str                    # LOW / NORMAL / ELEVATED / STRESS
    vol_percentile: float          # Where current vol sits in historical distribution

    def summary(self) -> str:
        lines = [
            f"  EWMA Vol Forecast     : {self.ewma_vol_forecast:.2%}",
            f"  Current Z-Score       : {self.zscore:.2f}  [{self.signal}]",
            f"  Vol Regime            : {self.regime}",
            f"  Vol Percentile        : {self.vol_percentile:.0f}th",
        ]
        return "\n".join(lines)


class VolatilityForecaster:
    """
    Forecasts near-term volatility and generates risk-aware signals.
    """

    def __init__(self, ewma_span: int = 30, zscore_window: int = 63, signal_threshold: float = 1.5):
        """
        Args:
            ewma_span:         Span (in days) for EWMA decay. Higher = slower to react.
            zscore_window:     Rolling window for z-score computation.
            signal_threshold:  Z-score magnitude to trigger LONG/SHORT signal.
        """
        self.ewma_span = ewma_span
        self.zscore_window = zscore_window
        self.signal_threshold = signal_threshold

    def forecast(self, df: pd.DataFrame, ticker: str = "") -> ForecastResult:
        """Run full forecasting pipeline and return a ForecastResult."""
        returns = df["returns"].dropna()
        prices = df["close"]

        ewma_vol = self._ewma_volatility(returns)
        vol_pct = self._vol_percentile(returns, ewma_vol)
        zscore = self._zscore(prices).iloc[-1]
        signal = self._signal(zscore)
        regime = self._regime(vol_pct)

        return ForecastResult(
            ticker=ticker,
            ewma_vol_forecast=ewma_vol,
            zscore=round(float(zscore), 3),
            signal=signal,
            regime=regime,
            vol_percentile=round(float(vol_pct), 1),
        )

    # ── Models ───────────────────────────────────────────────────────────────

    def _ewma_volatility(self, returns: pd.Series) -> float:
        """
        EWMA volatility: gives more weight to recent observations.
        Annualised via sqrt(252).
        """
        ewma_var = returns.ewm(span=self.ewma_span).var()
        latest_var = ewma_var.iloc[-1]
        return float(np.sqrt(latest_var * TRADING_DAYS))

    def ewma_vol_series(self, returns: pd.Series) -> pd.Series:
        """Full time series of EWMA annualised volatility."""
        return returns.ewm(span=self.ewma_span).std() * np.sqrt(TRADING_DAYS)

    def _zscore(self, prices: pd.Series) -> pd.Series:
        """
        Rolling z-score of price relative to its N-day mean.
        z = (price - rolling_mean) / rolling_std
        """
        rolling_mean = prices.rolling(self.zscore_window).mean()
        rolling_std = prices.rolling(self.zscore_window).std()
        return (prices - rolling_mean) / rolling_std

    def _signal(self, zscore: float) -> str:
        """
        Simple mean-reversion signal:
          z < -threshold → price is far below mean → potential LONG
          z >  threshold → price is far above mean → potential SHORT
          else           → NEUTRAL
        """
        if zscore < -self.signal_threshold:
            return "LONG"
        elif zscore > self.signal_threshold:
            return "SHORT"
        return "NEUTRAL"

    def _vol_percentile(self, returns: pd.Series, current_vol_ann: float) -> float:
        """Where does current EWMA vol sit in the distribution of all rolling vols?"""
        rolling_ann_vol = returns.rolling(self.ewma_span).std() * np.sqrt(TRADING_DAYS)
        rolling_ann_vol = rolling_ann_vol.dropna()
        if rolling_ann_vol.empty:
            return 50.0
        pct = (rolling_ann_vol < current_vol_ann).mean() * 100
        return float(pct)

    def _regime(self, vol_percentile: float) -> str:
        """Classify volatility regime based on its historical percentile."""
        if vol_percentile < 25:
            return "LOW"
        elif vol_percentile < 60:
            return "NORMAL"
        elif vol_percentile < 85:
            return "ELEVATED"
        return "STRESS"
