"""
Market data ingestion layer.

Fetches and normalises OHLCV data from Yahoo Finance.
Designed to be source-agnostic — swap out the provider
without touching downstream analysis code.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataFetcher:
    """
    Fetches historical OHLCV market data and normalises it into
    a consistent DataFrame schema for downstream analysis.
    """

    VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}

    def __init__(self, cache: bool = True):
        self.cache = cache
        self._store: dict[str, pd.DataFrame] = {}

    def fetch(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker:   Stock symbol, e.g. 'AAPL'
            period:   Lookback window, e.g. '1y', '6mo'
            interval: Bar size, e.g. '1d', '1h'

        Returns:
            DataFrame with columns: open, high, low, close, volume, returns
        """
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Invalid period '{period}'. Choose from: {self.VALID_PERIODS}")

        cache_key = f"{ticker}_{period}_{interval}"
        if self.cache and cache_key in self._store:
            logger.debug(f"Cache hit for {cache_key}")
            return self._store[cache_key]

        logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})")

        raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

        if raw.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

        df = self._normalise(raw, ticker)

        if self.cache:
            self._store[cache_key] = df

        logger.info(f"Ingested {len(df)} bars for {ticker}")
        return df

    def fetch_multiple(self, tickers: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers. Returns a dict keyed by ticker symbol.
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch(ticker, period=period, interval=interval)
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        return results

    @staticmethod
    def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Flatten multi-level columns, standardise names, add returns."""
        # yfinance sometimes returns MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "date"
        df["ticker"] = ticker
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = df["close"].apply(lambda x: x).pipe(
            lambda s: s.div(s.shift(1)).apply(lambda x: x).apply(
                lambda x: 0 if x <= 0 else __import__("math").log(x)
            )
        )
        df.dropna(inplace=True)
        return df

    def get_close_prices(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        """
        Returns a single DataFrame of close prices for multiple tickers.
        Useful for correlation and portfolio analysis.
        """
        frames = {}
        for ticker in tickers:
            try:
                df = self.fetch(ticker, period=period)
                frames[ticker] = df["close"]
            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")

        return pd.DataFrame(frames)
