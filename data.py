import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=60 * 60 * 6)  # cache for 6 hours
def load_price_data(symbols, start_date):
    df = yf.download(
        symbols,
        start=start_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(how="all")
    return df

def compute_cumulative_return(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices / prices.iloc[0] - 1) * 100

def compute_trailing_return(prices: pd.Series, trading_days: int = 252):
    """
    Compute trailing return over the last 'trading_days' trading days.
    252 ≈ 1 year of trading days
    """
    if len(prices) < trading_days:
        return None
    return (prices.iloc[-1] / prices.iloc[-trading_days] - 1) * 100
