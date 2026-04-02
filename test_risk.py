"""Unit tests for risk analysis module."""

import pytest
import numpy as np
import pandas as pd
from src.analysis.risk import RiskAnalyser
from src.analysis.forecast import VolatilityForecaster


def make_dummy_df(n=252, seed=42):
    """Generate synthetic price/returns data for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "close": prices,
        "returns": returns,
        "ticker": "TEST",
    }, index=pd.date_range("2023-01-01", periods=n, freq="B"))
    return df


class TestRiskAnalyser:
    analyser = RiskAnalyser(risk_free_rate=0.05)

    def test_annualised_vol_positive(self):
        df = make_dummy_df()
        vol = self.analyser.annualised_volatility(df["returns"])
        assert vol > 0

    def test_var_less_than_zero(self):
        df = make_dummy_df()
        var = self.analyser.var_historical(df["returns"], confidence=0.95)
        assert var < 0, "VaR should represent a loss (negative)"

    def test_cvar_leq_var(self):
        df = make_dummy_df()
        returns = df["returns"]
        var = self.analyser.var_historical(returns, 0.95)
        cvar = self.analyser.cvar(returns, 0.95)
        assert cvar <= var, "CVaR should be <= VaR (more extreme tail)"

    def test_max_drawdown_negative(self):
        df = make_dummy_df()
        dd = self.analyser.max_drawdown(df["close"])
        assert dd <= 0

    def test_full_report(self):
        df = make_dummy_df()
        report = self.analyser.analyse(df, ticker="TEST", period="1y")
        assert report.ticker == "TEST"
        assert report.n_observations == len(df) - 0  # returns already clean


class TestVolatilityForecaster:
    forecaster = VolatilityForecaster()

    def test_ewma_vol_positive(self):
        df = make_dummy_df()
        result = self.forecaster.forecast(df, ticker="TEST")
        assert result.ewma_vol_forecast > 0

    def test_signal_valid(self):
        df = make_dummy_df()
        result = self.forecaster.forecast(df, ticker="TEST")
        assert result.signal in {"LONG", "SHORT", "NEUTRAL"}

    def test_regime_valid(self):
        df = make_dummy_df()
        result = self.forecaster.forecast(df, ticker="TEST")
        assert result.regime in {"LOW", "NORMAL", "ELEVATED", "STRESS"}
