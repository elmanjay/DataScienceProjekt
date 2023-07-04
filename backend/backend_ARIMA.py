import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import yfinance as yf
import datetime

def predict_arima(df_input, years=3,prediction_days= 15):
    
    df = df_input.copy()
    df = df.dropna()
    df = df[["Date", "Close"]] # behalte nur Datum und Close Wert, verwerfe restlichen Spalten
    df["Date"] = pd.to_datetime(df["Date"]) # Konvertiere,d
    n = 360 * years # Nehme nur die letzten Jahre vom ges. Datensatz
    df = df.iloc[-n:]
    maxindex= max(df.index)
    listinex= list()
    for element in range(maxindex-10,maxindex+15):
        listinex.append(element)
    
    
    df["Train"] = df["Close"].iloc[:-10] 
    df["Test"] = df["Close"].iloc[-10:]
    new_df = pd.DataFrame(df["Test"]) # Erstelle Spalte 'Train' mit den Trainingsdaten
    

    model = sm.tsa.ARIMA(df["Train"] , order=(1, 2, 1))  # p, d, q sind die gewählten Ordnungsparameter
    model_fit = model.fit()
    

    # Prognose erstellen
    forecast = model_fit.forecast(steps=10+prediction_days) # Anzahl der zukünftigen Perioden für die Prognose angeben
    liste_forecasts = forecast.tolist()
    df_forecasts = pd.DataFrame(liste_forecasts , index=listinex)
    today = datetime.date.today()

    # Liste für die nächsten 14 Tage
    letzte_zeile = df.tail(1)
    gewünschter_wert = letzte_zeile.at[letzte_zeile.index[0], "Date"]
    date_list = [gewünschter_wert + timedelta(days=i) for i in range(1,15)]
    print(date_list)
    print(len(date_list))
    df_forecasts.rename(columns={0: "Prediction"}, inplace=True)
    merged_df = pd.merge(df, df_forecasts, left_index=True, right_index=True, how="outer")
    print(merged_df.tail(40))
    return(df_forecasts)



msft = yf.Ticker("NVD.DE")
df = msft.history(period="max")
df = df.reset_index()
result = predict_arima(df)