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


#Die Funktion windowed_df_to_date_X_y konvertiert einen fensterförmigen DataFrame in Eingabe-Features X und Zielwerte Y für das Training des LSTM-Modells.


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)

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

def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

 #Achtung optimale epochen für ablauf müssen noch bestimmt werden   
def lstm_stock_prediction(ticker_symbol, start_date, end_date, prediction_days=14):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date, actions=False)
    df = df[["Close"]]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

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

    model = create_model()
    train_model(model, x_train, y_train, x_train, y_train, epochs=30)

    x_test = np.array([test_data[-interval:, 0]])
    predictions = []

    for _ in range(prediction_days):
        prediction = model.predict(x_test)
        predictions.append(prediction[0][0])
        x_test = np.append(x_test[:, 1:], prediction, axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    last_date = df.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq="B")
    train_plot, test_plot= give_train_test(ticker_symbol, start_date, end_date)
    prediction_table = pd.DataFrame(predictions, columns=["Predicted Close"], index=prediction_dates)
    #plt.figure(figsize=(10, 6))
    #plt.plot(train_plot.index, train_plot["Close"], label="Train Data")
    #plt.plot(test_plot.index, test_plot["Close"], label="Test Data")
    #plt.plot(prediction_table.index, prediction_table["Predicted Close"], label="Vorhersage")
    #plt.xlabel("Date")
    #plt.ylabel("Closing Price")
    #plt.title("Stock Price Data")
    #plt.legend()
    #plt.show()
    #print(test_plot)
    return prediction_table , train_plot, test_plot

def give_train_test(ticker_symbol, start_date, end_date):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date, actions=False)
    df = df[["Close"]]
    train_len = int(len(df) * 0.92)
    train_data = df[:train_len]
    test_data = df[train_len - 2:]
    return train_data, test_data


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


prediction_table = lstm_stock_prediction("ALV.DE", "2020-05-25", "2023-06-25", prediction_days=14)
print(prediction_table)


