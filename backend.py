import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose(df):
    result = seasonal_decompose(df["Close"], model="additiv", period=12)
    df["Trend"] = decomp.trend
    df["Saison"] = decomp.seasonal
    df["Rauschen"] = decomp.resid
    return df