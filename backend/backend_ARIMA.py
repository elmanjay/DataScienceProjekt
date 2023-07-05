import pandas as pd
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def predict_arima(df_input, years=2,prediction_days= 15):

    def is_business_day(date):
        return bool(len(pd.bdate_range(date, date)))
    
    df = df_input.copy()
    df = df.dropna()
    df = df[["Date", "Close"]] # behalte nur Datum und Close Wert, verwerfe restlichen Spalten
    df["Date"] = pd.to_datetime(df["Date"]) # Konvertiere,d
    n = 360 * years # Nehme nur die letzten Jahre vom ges. Datensatz
    df = df.iloc[-n:]
    maxindex= max(df.index) # greift auf den max. Indexwert in einem Pandas Dataframe zurück
    listinex= list() # erstellen einer neuen leeren Liste mit dem Namen listinex
    for element in range(maxindex-10,maxindex+15): # Schleife erstellt Werte & fügt diese einer Liste hinzu, dbaei abhängig von max. Indexwert
        listinex.append(element)
    
    
    df["Train"] = df["Close"].iloc[:-10] 
    df["Test"] = df["Close"].iloc[-10:]
    #new_df = pd.DataFrame(df["Test"]) # Erstelle Spalte 'Train' mit den Trainingsdaten


    model = ARIMA(df["Train"] , order=(1, 2, 1))  # p, d, q sind die gewählten Ordnungsparameter
    model_fit = model.fit()
    

    # Prognose erstellen
    forecast = model_fit.forecast(steps=10+prediction_days) # Anzahl der zukünftigen Perioden für die Prognose angeben
    liste_forecasts = forecast.tolist() # erstellt eine Liste mit den Werten, die durch Model predicted wurden
    df_forecasts = pd.DataFrame(liste_forecasts , index=listinex) # erstellt ein df aus der Liste: liste_forecasts
    #today = datetime.date.today()

    # Liste für die nächsten 14 Tage
    letzte_zeile = df.tail(1) # speichert die letzte Zeile des urspr. df ab
    #gewünschter_wert = letzte_zeile.at[letzte_zeile.index[0], "Date"]
    #date_list = [gewünschter_wert + timedelta(days=i) for i in range(1,15)]
    #print(date_list)
    #print(len(date_list))
    df_forecasts.rename(columns={0: "Prediction"}, inplace=True)
    merged_df = pd.merge(df, df_forecasts, left_index=True, right_index=True, how="outer")

    #verschiebt die predicteten Werte um eins nach unten
    last_value = merged_df["Prediction"].iloc[-1] # Temporäre Variable für den letzten Wert der Spalte
    merged_df["Prediction"] = merged_df["Prediction"].shift(1) # den Rest der Spalte um eine Zeile nach unten verschieben
    merged_df.loc[merged_df.index[-1] + 1, "Prediction"] = last_value # letzten Wert in eine neue Zeile einfügen
    
    merged_df = merged_df.reset_index(drop=True)  # Setzt den Index des DataFrame zurück, um ihn neu zu nummerieren

    # Berechne den mse
    #error = mean_squared_error(df["Test"].iloc[-10:], df_forecasts.iloc[:10])
    #print(f"[Test evaluation] mean_squared_error: {error}")

    # Berechne den durchschnittlichen Zeitunterschied zwischen aufeinanderfolgenden Datenpunkten in der Spalte "Date"
    time_delta = merged_df["Date"].diff().mean().round("d") 
    print(f"Calculated avarage time delta rounded to days: {time_delta}")

    for index, date in enumerate(merged_df["Date"]):  # Iteriere über den Index und das Datum der Spalte "Date" des DataFrames
        if pd.isnull(date):  # Überprüfe, ob das Datum fehlt (null) ist
            previous_date = merged_df.at[index-1, "Date"]  # Speichere das vorherige Datum
            final_date = previous_date + time_delta # Berechne das nächste Datum um `time_delta` erhöht

            while not is_business_day(final_date):  # Überprüfe, ob das berechnete Datum ein Werktag ist
                final_date = final_date + pd.Timedelta(days=1)  # Falls nicht, erhöhe das Datum um `time_delta` und überprüfe erneut

            merged_df.at[index, "Date"] = final_date  # Setze das berechnete Datum als neues Datum für die aktuelle Zeile im DataFrame
            
    print(merged_df.tail(40))
    return(df_forecasts)



msft = yf.Ticker("NVD.DE")
df = msft.history(period="max")
df = df.reset_index()
result = predict_arima(df)
