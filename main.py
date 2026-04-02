"""
AlgoRisk — CLI entry point.

Usage:
    python main.py --ticker AAPL --period 1y
    python main.py --ticker AAPL MSFT TSLA --period 6mo
"""

import argparse
from src.ingestion.fetcher import MarketDataFetcher
from src.analysis.risk import RiskAnalyser
from src.analysis.forecast import VolatilityForecaster
from src.utils.logger import get_logger

logger = get_logger("algorisk")


def run(tickers: list[str], period: str):
    fetcher = MarketDataFetcher()
    analyser = RiskAnalyser(risk_free_rate=0.05)
    forecaster = VolatilityForecaster()

    for ticker in tickers:
        print(f"\n[AlgoRisk] Fetching data for {ticker} (period={period})...")
        try:
            df = fetcher.fetch(ticker, period=period)
            print(f"[AlgoRisk] Data ingested: {len(df)} trading days\n")

            report = analyser.analyse(df, ticker=ticker, period=period)
            print(report.summary())

            forecast = forecaster.forecast(df, ticker=ticker)
            print(forecast.summary())

        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description="AlgoRisk — Market Risk Analyser")
    parser.add_argument("--ticker", nargs="+", required=True, help="One or more ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--period", default="1y", help="Lookback period (default: 1y)")
    args = parser.parse_args()

    run(tickers=[t.upper() for t in args.ticker], period=args.period)


if __name__ == "__main__":
    main()
