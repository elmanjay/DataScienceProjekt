import pandas as pd
import numpy as np
import math 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import datetime
import locale
import pytz

#Startjahr kann als String mit übergeben werden
def make_pred(df, startjahr):
    df.reset_index(inplace=True)
    df["Date"] = pd.Series(df["Date"], dtype="string")
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    df = df.loc[:, ["Date", "Close"]]
    start_year = int(startjahr)  # Das gewünschte Startjah
    df = df[df["Date"].str[:4].astype(int) >= start_year]
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train["Close"]
    X_test = np.array(test.index).reshape(-1, 1)
    y_test = test["Close"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_all = np.array(df.index).reshape(-1, 1)
    predictions = model.predict(X_all)
    predictionstest = model.predict(X_test)
    predictionstrain = model.predict(X_train)
    df["Predictions"] = predictions
    df["Predictions"] = predictions
    df["Train"] = np.where(df.index.isin(train.index), df["Close"], np.nan)
    df["Test"] = np.where(df.index.isin(test.index), df["Close"], np.nan)
    r2_score_result = r2_score(y_train, predictionstrain)
    df["R2 Score"] = r2_score_result
    mse = mean_squared_error(y_test, predictionstest)
    df["MSE"] = mse
    mae = mean_absolute_error(y_test, predictionstest)
    df["MAE"] = mae
    rmse = np.sqrt(mse)
    df["RMSE"] = rmse
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=14, freq='D')
    next_14_values = np.arange(X_test.max()+ 1, X_test.max() + 15).reshape(-1, 1)
    future_predictions= model.predict(next_14_values)
    today = datetime.date.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, 15)]
    future_df = pd.DataFrame({
    "Date": future_dates,
    "Predictions": future_predictions
            })
    return df, future_df


def make_pred_month(df, daysgiven):
    df = df.copy()  # Erstelle eine Kopie des DataFrames, um Änderungen daran vorzunehmen
    df = df.drop(["Open", "High","Low","Volume","Dividends","Stock Splits"], axis=1)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  # Konvertiere das Datum in das richtige Format
    now = datetime.datetime.now(pytz.timezone('America/New_York'))  # Aktuelles Datum und Uhrzeit in der Zeitzone New York
    zeitpunkt = now - datetime.timedelta(days=daysgiven)  # Berechne den Zeitpunkt basierend auf der angegebenen Anzahl von Tagen
    zeitpunktformat = np.datetime64(zeitpunkt)  # Konvertiere den Zeitpunkt in das richtige Format
    df = df.loc[df["Date"] >= zeitpunktformat]  # Filtere den DataFrame nach dem Zeitpunkt
    df["Date"] = pd.Series(df["Date"], dtype="string")  # Konvertiere das Datum in einen String
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')  # Extrahiere das Datum im Format 'YYYY-MM-DD'
    train, test = train_test_split(df, test_size=0.2, random_state=42)  # Teile den DataFrame in Trainings- und Testdaten auf
    X_train = np.array(train.index).reshape(-1, 1)  # Extrahiere die Indizes der Trainingsdaten
    y_train = train["Close"]  # Extrahiere die Zielvariablen der Trainingsdaten
    X_test = np.array(test.index).reshape(-1, 1)  # Extrahiere die Indizes der Testdaten
    y_test = test["Close"]  # Extrahiere die Zielvariablen der Testdaten
    model = LinearRegression()  # Initialisiere das lineare Regressionsmodell
    model.fit(X_train, y_train)  # Trainiere das Modell mit den Trainingsdaten
    X_all = np.array(df.index).reshape(-1, 1)  # Extrahiere alle Indizes
    predictions = model.predict(X_all)  # Mache Vorhersagen für alle Datenpunkte
    predictionstest = model.predict(X_test)  # Mache Vorhersagen für die Testdaten
    predictionstrain = model.predict(X_train)  # Mache Vorhersagen für die Trainingsdaten
    df["Predictions"] = predictions  # Füge die Vorhersagen dem DataFrame hinzu
    df.loc[df.index.isin(train.index), "Train"] = df.loc[df.index.isin(train.index), "Close"].copy()  # Füge die Trainingsdaten dem DataFrame hinzu
    df.loc[df.index.isin(test.index), "Test"] = df.loc[df.index.isin(test.index), "Close"].copy()  # Füge die Testdaten dem DataFrame hinzu
    r2_score_result = r2_score(y_train, predictionstrain)  # Berechne den R2-Score für die Trainingsdaten
    df.loc[:, "R2 Score"] = r2_score_result  # Füge den R2-Score dem DataFrame hinzu
    mse = mean_squared_error(y_test, predictionstest)  # Berechne den Mean Squared Error (MSE) für die Testdaten
    df.loc[:, "MSE"] = mse  # Füge den MSE dem DataFrame hinzu
    mae = mean_absolute_error(y_test, predictionstest)  # Berechne den Mean Absolute Error (MAE) für die Testdaten
    df.loc[:, "MAE"] = mae  # Füge den MAE dem DataFrame hinzu
    rmse = np.sqrt(mse)  # Berechne die Root Mean Squared Error (RMSE) für die Testdaten
    df.loc[:, "RMSE"] = rmse  # Füge den RMSE dem DataFrame hinzu
    scaled_mae = mae / test["Close"].mean()  # Berechne den skalierten MAE für die Testdaten
    df.loc[:, "Scaled_mae"] = scaled_mae  # Füge den skalierten MAE dem DataFrame hinzu
    last_date = pd.to_datetime(df["Date"].iloc[-1])  # Extrahiere das letzte Datum im DataFrame
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=14, freq='D')  # Erzeuge zukünftige Datumsangaben
    next_14_values = np.arange(X_test.max() + 1, X_test.max() + 15).reshape(-1, 1)  # Berechne die Indizes für die zukünftigen Vorhersagen
    future_predictions = model.predict(next_14_values)  # Mache zukünftige Vorhersagen
    today = datetime.date.today()  # Aktuelles Datum
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, 15)]  # Erzeuge die Datumsangaben für die nächsten 14 Tage
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predictions": future_predictions
    })  # Erzeuge ein DataFrame für die zukünftigen Vorhersagen
    print(df.head())
    return df, future_df  # Gib den aktuellen DataFrame und das zukünftige DataFrame zurück




