import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima

path = "/Users/silasmock/Desktop/NVDA.csv"
 # read the csv file
df = pd.read_csv(path)
df = df.dropna() # entferne Einträge mit NaN

def predict_arima(df, years=7):
    
    
    df = df[["Date", "Close"]] # behalte nur Datum und Close Wert, verwerfe restlichen Spalten
    df['Date'] = pd.to_datetime(df["Date"]) # Konvertiere,d
    n = 360 * years # Nehme nur die letzten Jahre vom ges. Datensatz
    df = df.iloc[-n:]
    
    df["Train"] = df["Close"].iloc[:-10]  # Erstelle Spalte 'Train' mit den Trainingsdaten
    df["Test"] = df["Close"].iloc[-10:]  # Erstelle Spalte 'Test' mit den Testdaten
    
    
    # Fitte ein Arima Modell auf den Trainings-Datensatz
    train = df.iloc[:-10, 1]  # alle Datenpunkte bis auf die letzten 10 zum trainieren
    test = df.iloc[-10:, 1]  # die letzten 10 zum testen des Modells
    model = auto_arima(train, trace=True, error_action="ignore", suppress_warnings=True) # verwenden der Funktion Autoarima, trace=true gibt diagnostische Informationen des Modells an
    model.fit(train)
    

    
    # hier werden die Predictions für die Testdaten gemacht -> das Modell sagt von dem letzten Trainingsdatensatz die nächsten 10 vorher
    test_forecast = model.predict(n_periods=10)
    test_forecast = pd.DataFrame(test_forecast, index=test.index, columns=["Predictions"])

    # führe Spalte mit Predictions auf Testwerte, mit restlichem Dataframe zusammen
    df = pd.concat([df, test_forecast], axis=1)
    
    # nehmen gesamten DataFrame und trainieren mit diesem ein erneutes Modell
    model = auto_arima(df["Close"], trace=True, error_action="ignore", suppress_warnings=True)
    model.fit(df["Close"])
    future_forecast = model.predict(n_periods=10) # 10 Vorhersagen in die Zukunft, wo Werte noch unbekannt

    p, d, q = model.order # speichern Parameter des Arima Modells ab
    
    # erstellen DataFrame, mit Vorhersagen & die nächsten 10 Business Tage als Index, damit wir diesen df an anderen anhängen
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=10, freq="B").date
    Predictions = pd.DataFrame({"Date": future_dates, "Predictions": future_forecast})
    
    
    df = pd.concat([df, Predictions], ignore_index=True) # führen hier die zwei erstellten df zusammen
    df["Date"] = pd.to_datetime(df["Date"]) # bevor wir df zurück geben, koveertieren Date in datetime 
    df.set_index("Date", inplace=True)
    
    
    
    return df, p, d, q

df,p, d, q = predict_arima(df)


pd.set_option("display.max_rows", None)
result = predict_arima(df)
print(result)