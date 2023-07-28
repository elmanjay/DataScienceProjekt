import pandas as pd
import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Funktion, die als Input die ungefilterten Daten aus dem dcc.Store erhält, die Wahl der Hyperparameter (p, d, q) 
#und das relevante Zeitintervall der Datengrundlage in Jahren sowie den gewünschten Prognosehorizont.
#Ausgabe: DataFrame mit Trainings-, Test- und Prognosedaten sowie Metriken.
def predict_arima(df_input, p, d, q, years=2, prediction_days=15):
    # Hilfsfunktion, um zu überprüfen, ob ein Datum ein Arbeitstag ist
    def is_business_day(date):
        return bool(len(pd.bdate_range(date, date)))
    
# Formatieren und Filtern des DataFrames
    df = df_input.copy()
    df = df.dropna()
    df = df[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    n = 360 * years
    df = df.iloc[-n:]
    maxindex = max(df.index)
    listinex = list()
    for element in range(maxindex - 10, maxindex + 15):
        listinex.append(element)
    
    # Teile die "Close" Spalte in einen Trainings- und einen Testdatensatz auf
    df["Train"] = df["Close"].iloc[:-10]
    df["Test"] = df["Close"].iloc[-10:]
    
    # Erstelle ein ARIMA-Modell mit den angegebenen Ordnungsparametern p, d und q
    model = ARIMA(df["Train"], order=(p, d, q))
    # Führe die Anpassung des Modells an die Daten durch
    model_fit = model.fit()
    
    # Erstelle eine Prognose für die nächsten 10+`prediction_days` Perioden
    forecast = model_fit.forecast(steps=10 + prediction_days)
    # Konvertiere die Prognose in eine Liste
    liste_forecasts = forecast.tolist()
    # Erstelle ein DataFrame mit den Prognosewerten und den entsprechenden Indizes
    df_forecasts = pd.DataFrame(liste_forecasts, index=listinex)
    
    # Verschiebe die prognostizierten Werte um eine Zeile nach unten
    last_value = df_forecasts.iloc[-1, 0]
    df_forecasts[0] = df_forecasts[0].shift(1)
    df_forecasts.loc[df_forecasts.index[-1] + 1, 0] = last_value
    
    # Führe einen äußeren Join der ursprünglichen Daten und der Prognosen durch
    df_wholedata = pd.merge(df, df_forecasts, left_index=True, right_index=True, how="outer")
    # Setze den Index des DataFrame zurück, um ihn neu zu nummerieren
    df_wholedata = df_wholedata.reset_index(drop=True)

    # Berechne verschiedene Metriken, um die Leistung des Modells zu bewerten
    mae = mean_absolute_error(df["Test"].iloc[-10:], df_forecasts.iloc[:10])
    mse = mean_squared_error(df["Test"].iloc[-10:], df_forecasts.iloc[:10])
    rmse = np.sqrt(mse)
    scaled_mae = mae / df["Test"].iloc[-10:].mean()
    metrics = [mae, mse, rmse, scaled_mae]
    
    # Berechne den durchschnittlichen Zeitunterschied zwischen aufeinanderfolgenden Datenpunkten in der "Date" Spalte
    time_delta = df_wholedata["Date"].diff().mean().round("d") 

    # Iteriere über die "Date" Spalte, um fehlende Datumsangaben zu vervollständigen
    for index, date in enumerate(df_wholedata["Date"]):
        if pd.isnull(date):
            # Speichere das vorherige Datum
            previous_date = df_wholedata.at[index - 1, "Date"]
            # Berechne das nächste Datum um `time_delta` erhöht
            final_date = previous_date + time_delta

            # Überprüfe, ob das berechnete Datum ein Arbeitstag ist
            while not is_business_day(final_date):
                # Falls nicht, erhöhe das Datum um `time_delta` und überprüfe erneut
                final_date = final_date + pd.Timedelta(days=1)

            # Setze das berechnete Datum als neues Datum für die aktuelle Zeile im DataFrame
            df_wholedata.at[index, "Date"] = final_date

    # Gib das DataFrame mit den Prognosen und die berechneten Metriken zurück
    return df_wholedata, metrics



