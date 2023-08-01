import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
import datetime
import pytz


#Die Funktion konvertiert einen fensterförmigen DataFrame in Eingabe-Features X und Zielwerte Y für das Training des LSTM-Modells.
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

#Die Funktion definiert die Gestalt des neuronalen Netzes.
def create_model():
    model = Sequential([
        layers.Input((10, 1)),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mean_absolute_error"])
    return model

#Die Funktion Trainiert das neuronale Netz auf den übergebenen Daten.
#Überwachung währenddessen erfolgt mit den Validierungsdatensätzen X_val, y_val, Modell trainiert über Zeitraum von 200 Epochen.
def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

#Die Funktion erhält die Daten der Aktie sowie das gewünschte Zeitintervall, um die Modelle vorzutrainieren und zu speichern.
#Nachdem die Daten zurücktransformiert wurden, gibt es drei Data Frames: 
#future_prediction_table, test_prediction_table und full_table, die die Vorhersagen und die tatsächlichen Werte enthalten.
def lstm_stock_prediction_pretrain(df, daysgiven, prediction_days=14, ticker="Default"):
    df = df.copy() 
    #Bearbeitung der Datengrundlage 
    df = df.drop(["Open", "High","Low","Volume","Dividends","Stock Splits"], axis=1)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  
    now = datetime.datetime.now(pytz.timezone('America/New_York'))  
    zeitpunkt = now - datetime.timedelta(days=daysgiven)  
    zeitpunktformat = np.datetime64(zeitpunkt)  
    df = df.loc[df["Date"] >= zeitpunktformat]  
    df["Date"] = pd.Series(df["Date"], dtype="string")  
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})') 
    df = df.set_index("Date")
    #Skalieren der Daten
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    #Aufteilen in Trainings und Testdaten
    train_len = int(len(scaled_data) * 0.92)
    train_data = scaled_data[:train_len, :]
    test_data = scaled_data[train_len - 2:, :]

    x_train, y_train = [], []
    interval = 10

    for i in range(interval, len(train_data)):
        x_train.append(train_data[i - interval:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Speichern des Modells 
    model = create_model()
    train_model(model, x_train, y_train, x_train, y_train, epochs=200)
    name = str(ticker)+ "_lstm_model" 
    save_model(model, "models/lstm/"+str(name), save_format='tf')

#Funktion zur Erstellung der Prognose. Sie erhält die Daten sowie das gewünschte Zeitintervall.
#Anschließend wird das benötigte Modell geladen und eine Prognose für die gewünschten Tage erstellt.
def lstm_stock_prediction(df, daysgiven, ticker="ALV.DE", prediction_days=14):
    df = df.copy()  
    # Datenbereinigung
    df = df.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  
    now = datetime.datetime.now(pytz.timezone("America/New_York"))  
    zeitpunkt = now - datetime.timedelta(days=daysgiven)  
    zeitpunktformat = np.datetime64(zeitpunkt)  
    df = df.loc[df["Date"] >= zeitpunktformat]  
    df["Date"] = pd.Series(df["Date"], dtype="string")  
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})') 
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    train_len = int(len(scaled_data) * 0.8)
    test_cutoff = train_len
    train_data = scaled_data[:train_len, :]
    test_data = scaled_data[train_len - 2:, :]

    x_train, y_train = [], []
    interval = 10

    # Erstellen der Trainingsdaten mit einer Fenstergröße von 'interval'
    for i in range(interval, len(train_data)):
        x_train.append(train_data[i - interval:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Laden des richtigen Modells
    name = str(ticker) + "_lstm_model" 
    model = load_model("models/lstm/" + str(name))

    test_predictions = []

    full_data_intervalled = []

    # Vorhersage für den Testdatenbereich mit Fenstern der Größe 'interval'
    for i in range(test_cutoff, len(scaled_data)):
        full_data_intervalled.append(scaled_data[i - interval:i, 0])

    x_intervalled = np.array(full_data_intervalled)
    x_intervalled = np.reshape(x_intervalled, (x_intervalled.shape[0], x_intervalled.shape[1], 1))

    for i in range(len(full_data_intervalled)):
        x = x_intervalled[i]
        x = x.T
        prediction = model.predict(x)
        test_predictions.append(prediction[0][0])

    x_test = np.array([test_data[-interval:, 0]])
    predictions = []

    # Vorhersage für das gewünschte Zeitintervall mit Fenstern der Größe 'interval'
    for _ in range(prediction_days):
        prediction = model.predict(x_test)
        predictions.append(prediction[0][0])
        x_test = np.append(x_test[:, 1:], prediction, axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

    first_date = df.index[test_cutoff]
    last_date = df.index[-1]
    all_dates = df.index[:]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq="B")
    test_predictions_dates = df.index[test_cutoff:test_cutoff + test_predictions.shape[0]]

    # Erstellen der Ergebnistabellen
    future_prediction_table = pd.DataFrame(predictions, columns=["Predicted Future"], index=prediction_dates)

    test_prediction_table = pd.DataFrame(test_predictions, columns=["Predicted Close"], index=test_predictions_dates)
    test_prediction_table["True Close"] = df["Close"].values[test_cutoff:test_predictions.shape[0] + test_cutoff]

    full_table = pd.DataFrame(index=all_dates)
    full_table["Train"] = np.nan
    full_table["Test"] = np.nan
    full_table["Predicted Test"] = np.nan
    cutoff = test_prediction_table.head(int(len(test_prediction_table) * 0.9)).index[-1]
    full_table.loc[full_table.index < cutoff, "Train"] = df[df.index < cutoff]["Close"]
    full_table.loc[full_table.index >= cutoff, "Test"] = df[df.index >= cutoff]["Close"]
    full_table.loc[full_table.index >= cutoff, "Predicted Test"] = test_prediction_table[test_prediction_table.index >= cutoff]["Predicted Close"]

    full_table = pd.concat([full_table, future_prediction_table])
    metrics_table = calculate_metrics(test_prediction_table)

    return full_table, metrics_table, future_prediction_table


#Funktion zur Berechnung verschiedener Metriken für die Vorhersage
 #Die berechneten Metriken werden in dem neuen Data Frame metrics_table gespeichert. 
def calculate_metrics(prediction_table):
    true_values = prediction_table["True Close"]
    predicted_values = prediction_table["Predicted Close"]

    mse = np.mean((true_values - predicted_values) ** 2)
    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(mse)
    scaled_mae = mae / np.mean(true_values)

    metrics_table = pd.DataFrame({
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "Scaled MAE": [scaled_mae]
    })

    return metrics_table

#Funktion zum Vorab-Training mehrerer Modelle für verschiedene Ticker
def pretrain_list(list_ticker):
    for element in list_ticker:
        msft = yf.Ticker(element)
        df = msft.history(period="max")
        df.reset_index(inplace= True)
        lstm_stock_prediction_pretrain(df, 1095, prediction_days=14, ticker=element)



#assets = ["ALV.DE", "AMZ.DE", "DPW.DE", "MDO.DE", "NVD.DE","^MDAXI"]
#pretrain_list(assets)




