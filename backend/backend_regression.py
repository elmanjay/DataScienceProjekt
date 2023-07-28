import pandas as pd
import numpy as np
import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Funktion, die als Input die ungefilterten Daten aus dem dcc.Store erhält und das relevante Zeitintervall in Tagen.
#Ausgabe: DataFrame mit Trainings-, Test- und Prognosedaten für die nächsten 14 Tage sowie die berechneten Metriken.
def make_pred_reg(df, days_to_consider):
    # Formatieren und Filtern des DataFrames
    df_wholedata = df.copy()
    df_wholedata = df_wholedata.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df_wholedata["Date"] = pd.to_datetime(df_wholedata["Date"]).dt.tz_localize(None)
    
    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    start_date = now - datetime.timedelta(days=days_to_consider)
    start_date_np = np.datetime64(start_date)
    
    df_wholedata = df_wholedata.loc[df_wholedata["Date"] >= start_date_np]
    df_wholedata["Date"] = pd.Series(df_wholedata["Date"], dtype="string")
    df_wholedata["Date"] = df_wholedata["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    
    # Aufteilen in  Trainings und Testdaten
    train_df, test_df = train_test_split(df_wholedata, test_size=0.2, random_state=42)
    
    X_train = np.array(train_df.index).reshape(-1, 1)
    y_train = train_df["Close"]
    
    X_test = np.array(test_df.index).reshape(-1, 1)
    y_test = test_df["Close"]
    
    # Erstellen des Modells
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Erstellen der Test-Vorhersage
    X_all = np.array(df_wholedata.index).reshape(-1, 1)
    predictions_all = model.predict(X_all)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    
    df_wholedata["Predictions"] = predictions_all
    df_wholedata.loc[df_wholedata.index.isin(train_df.index), "Train"] = df_wholedata.loc[df_wholedata.index.isin(train_df.index), "Close"].copy()
    df_wholedata.loc[df_wholedata.index.isin(test_df.index), "Test"] = df_wholedata.loc[df_wholedata.index.isin(test_df.index), "Close"].copy()
    
    # Berechnung der Performancemaße und einfügen in den DF
    r2_score_train = r2_score(y_train, predictions_train)
    df_wholedata.loc[:, "R2 Score"] = r2_score_train
    
    mse_test = mean_squared_error(y_test, predictions_test)
    df_wholedata.loc[:, "MSE"] = mse_test
    
    mae_test = mean_absolute_error(y_test, predictions_test)
    df_wholedata.loc[:, "MAE"] = mae_test
    
    rmse_test = np.sqrt(mse_test)
    df_wholedata.loc[:, "RMSE"] = rmse_test
    
    scaled_mae_test = mae_test / test_df["Close"].mean()
    df_wholedata.loc[:, "Scaled MAE"] = scaled_mae_test
    
    # Erstellen der Zukunftsprognose
    last_date = pd.to_datetime(df_wholedata["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=14, freq='D')
    next_14_values = np.arange(df_wholedata.index[-1] + 1, df_wholedata.index[-1] + 15).reshape(-1, 1)
    future_predictions = model.predict(next_14_values)
    
    today = datetime.date.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, 15)]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predictions": future_predictions
    })
    
    return df_wholedata, future_df







