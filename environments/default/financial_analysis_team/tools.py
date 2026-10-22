# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock financial data feed for the financial_analysis_team MAS."""

import json

_MARKET = {
    "AAPL": {"price": 192.34, "volume": 58_200_000, "market_cap": 2_990_000_000_000},
    "MSFT": {"price": 415.18, "volume": 21_800_000, "market_cap": 3_080_000_000_000},
    "NVDA": {"price": 138.07, "volume": 178_400_000, "market_cap": 3_400_000_000_000},
    "TSLA": {"price": 248.42, "volume": 96_700_000, "market_cap": 793_000_000_000},
    "GOOGL": {"price": 167.81, "volume": 17_500_000, "market_cap": 2_080_000_000_000},
}

_FUNDAMENTALS = {
    "AAPL": {"pe": 31.2, "revenue": 383_300_000_000, "earnings": 96_995_000_000},
    "MSFT": {"pe": 36.1, "revenue": 245_122_000_000, "earnings": 88_136_000_000},
    "NVDA": {"pe": 64.7, "revenue": 96_307_000_000, "earnings": 53_008_000_000},
    "TSLA": {"pe": 67.4, "revenue": 96_773_000_000, "earnings": 14_997_000_000},
    "GOOGL": {"pe": 25.9, "revenue": 307_394_000_000, "earnings": 73_795_000_000},
}

_TECHNICAL = {
    "AAPL": {"rsi": 58.2, "macd": "bullish", "ma_50": 188.45},
    "MSFT": {"rsi": 49.7, "macd": "neutral", "ma_50": 412.10},
    "NVDA": {"rsi": 71.4, "macd": "overbought", "ma_50": 132.88},
    "TSLA": {"rsi": 38.5, "macd": "bearish", "ma_50": 263.71},
    "GOOGL": {"rsi": 54.0, "macd": "bullish", "ma_50": 162.30},
}

_NEWS = {
    "AAPL": {"sentiment": 0.42, "recent": ["AAPL beats Q3 estimates", "Vision Pro EU launch confirmed"]},
    "TSLA": {"sentiment": -0.18, "recent": ["TSLA delivery miss in Q3", "Cybertruck recall"]},
    "NVDA": {"sentiment": 0.71, "recent": ["NVDA H200 sells out through 2025", "Blackwell roadmap accelerated"]},
}

_PORTFOLIOS = {
    "PORT-1": {"holdings": ["AAPL", "MSFT", "NVDA"], "allocation": {"AAPL": 0.40, "MSFT": 0.35, "NVDA": 0.25}},
    "PORT-2": {"holdings": ["TSLA", "GOOGL"], "allocation": {"TSLA": 0.55, "GOOGL": 0.45}},
}

_RISK = {
    "AAPL": {"volatility": 0.21, "beta": 1.18, "var_95": 0.024},
    "MSFT": {"volatility": 0.19, "beta": 0.94, "var_95": 0.022},
    "NVDA": {"volatility": 0.42, "beta": 1.71, "var_95": 0.051},
    "TSLA": {"volatility": 0.55, "beta": 2.04, "var_95": 0.067},
    "GOOGL": {"volatility": 0.23, "beta": 1.05, "var_95": 0.027},
}


def get_stock_price(ticker: str) -> str:
    """Return the current price for a ticker."""
    m = _MARKET.get(ticker)
    if not m:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "price": m["price"]})


def get_stock_volume(ticker: str) -> str:
    """Return the current trading volume for a ticker."""
    m = _MARKET.get(ticker)
    if not m:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "volume": m["volume"]})


def get_market_cap(ticker: str) -> str:
    """Return the market capitalization for a ticker."""
    m = _MARKET.get(ticker)
    if not m:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "market_cap": m["market_cap"]})


def get_pe_ratio(ticker: str) -> str:
    """Return the P/E ratio for a ticker."""
    f = _FUNDAMENTALS.get(ticker)
    if not f:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "pe": f["pe"]})


def get_revenue(ticker: str) -> str:
    """Return the latest annual revenue for a ticker."""
    f = _FUNDAMENTALS.get(ticker)
    if not f:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "revenue": f["revenue"]})


def get_earnings(ticker: str) -> str:
    """Return the latest annual earnings for a ticker."""
    f = _FUNDAMENTALS.get(ticker)
    if not f:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "earnings": f["earnings"]})


def get_rsi(ticker: str) -> str:
    """Return the 14-day RSI for a ticker."""
    t = _TECHNICAL.get(ticker)
    if not t:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "rsi": t["rsi"]})


def get_macd(ticker: str) -> str:
    """Return the MACD signal for a ticker."""
    t = _TECHNICAL.get(ticker)
    if not t:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "macd": t["macd"]})


def get_moving_average(ticker: str) -> str:
    """Return the 50-day moving average for a ticker."""
    t = _TECHNICAL.get(ticker)
    if not t:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "ma_50": t["ma_50"]})


def get_sentiment_score(ticker: str) -> str:
    """Return aggregate news sentiment for a ticker."""
    n = _NEWS.get(ticker)
    if not n:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "sentiment": n["sentiment"]})


def get_recent_news(ticker: str) -> str:
    """Return latest news headlines for a ticker."""
    n = _NEWS.get(ticker)
    if not n:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "recent": n["recent"]})


def get_portfolio_holdings(portfolio_id: str) -> str:
    """Return the holdings list of a portfolio."""
    p = _PORTFOLIOS.get(portfolio_id)
    if not p:
        return json.dumps({"error": f"unknown portfolio_id {portfolio_id}"})
    return json.dumps({"portfolio_id": portfolio_id, "holdings": p["holdings"]})


def get_portfolio_allocation(portfolio_id: str) -> str:
    """Return percent allocation per asset for a portfolio."""
    p = _PORTFOLIOS.get(portfolio_id)
    if not p:
        return json.dumps({"error": f"unknown portfolio_id {portfolio_id}"})
    return json.dumps({"portfolio_id": portfolio_id, "allocation": p["allocation"]})


def get_volatility(ticker: str) -> str:
    """Return 30-day annualized volatility for a ticker."""
    r = _RISK.get(ticker)
    if not r:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "volatility": r["volatility"]})


def get_beta(ticker: str) -> str:
    """Return the market beta for a ticker."""
    r = _RISK.get(ticker)
    if not r:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "beta": r["beta"]})


def get_var(ticker: str) -> str:
    """Return the 1-day 95% Value-at-Risk for a ticker."""
    r = _RISK.get(ticker)
    if not r:
        return json.dumps({"error": f"unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, "var_95": r["var_95"]})
