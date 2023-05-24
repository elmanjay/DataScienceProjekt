import pandas as pd
import numpy as np
import math 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.linear_model import LinearRegression
import datetime

#Startjahr kann als String mit übergeben werden
def make_plot(df, startjahr):
    df.reset_index(inplace=True)
    df["Date"] = pd.Series(df["Date"], dtype="string")
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    df = df.loc[:, ["Date", "Close"]]
    start_year = int(startjahr)  # Das gewünschte Startjah
    df = df[df["Date"].str[:4].astype(int) >= start_year]

    #train, test = train_test_split(df, test_size=0.20)

    train_size = int(len(df) * 0.8)  # 80% der Daten für das Training
    train = df[:train_size]
    test = df[train_size:]
    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train["Close"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_all = np.array(df.index).reshape(-1, 1)
    predictions = model.predict(X_all)
    df["Predictions"] = predictions
    df["Predictions"] = predictions
    df["Train"] = np.where(df.index.isin(train.index), df["Close"], np.nan)
    df["Test"] = np.where(df.index.isin(test.index), df["Close"], np.nan)
    return df