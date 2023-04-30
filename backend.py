import pandas as pd
import statsmodels.api as sm

def decompose(df):
    decomposition = sm.tsa.seasonal_decompose(df["Close"], model="multiplikativ", period=12)
    df["Trend"] = decomposition.trend
    df["Saison"] = decomposition.seasonal
    df["Rauschen"] = decomposition.resid
    return df