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