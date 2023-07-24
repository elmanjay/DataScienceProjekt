import pandas as pd
import numpy as np
import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Funktion die als Input die ungefilterten Daten aus dem dcc.Store erählt und das relevante zeitintervall in Tage
#Ausgabe: Dataframe mit Trainigs-,Test- und Prognosedaten 
def make_pred_reg(df, days_to_consider):
    #Formatieren und Filtern des Data Frames
    df_copy = df.copy()
    df_copy = df_copy.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df_copy["Date"] = pd.to_datetime(df_copy["Date"]).dt.tz_localize(None)
    
    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    start_date = now - datetime.timedelta(days=days_to_consider)
    start_date_np = np.datetime64(start_date)
    
    df_copy = df_copy.loc[df_copy["Date"] >= start_date_np]
    df_copy["Date"] = pd.Series(df_copy["Date"], dtype="string")
    df_copy["Date"] = df_copy["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    
    #Aufteilen in  Trainings und Testdaten
    train_df, test_df = train_test_split(df_copy, test_size=0.2, random_state=42)
    
    X_train = np.array(train_df.index).reshape(-1, 1)
    y_train = train_df["Close"]
    
    X_test = np.array(test_df.index).reshape(-1, 1)
    y_test = test_df["Close"]
    
    #Erstellen des Modells
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #Ersellen der Test-Vorhersage
    X_all = np.array(df_copy.index).reshape(-1, 1)
    predictions_all = model.predict(X_all)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    
    df_copy["Predictions"] = predictions_all
    df_copy.loc[df_copy.index.isin(train_df.index), "Train"] = df_copy.loc[df_copy.index.isin(train_df.index), "Close"].copy()
    df_copy.loc[df_copy.index.isin(test_df.index), "Test"] = df_copy.loc[df_copy.index.isin(test_df.index), "Close"].copy()
    
    #Berechnung der Performancemaße und einfügen in den DF
    r2_score_train = r2_score(y_train, predictions_train)
    df_copy.loc[:, "R2 Score"] = r2_score_train
    
    mse_test = mean_squared_error(y_test, predictions_test)
    df_copy.loc[:, "MSE"] = mse_test
    
    mae_test = mean_absolute_error(y_test, predictions_test)
    df_copy.loc[:, "MAE"] = mae_test
    
    rmse_test = np.sqrt(mse_test)
    df_copy.loc[:, "RMSE"] = rmse_test
    
    scaled_mae_test = mae_test / test_df["Close"].mean()
    df_copy.loc[:, "Scaled MAE"] = scaled_mae_test
    
    #Ersellen der Zukunftsprognose
    last_date = pd.to_datetime(df_copy["Date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=14, freq='D')
    next_14_values = np.arange(df_copy.index[-1] + 1, df_copy.index[-1] + 15).reshape(-1, 1)
    future_predictions = model.predict(next_14_values)
    
    today = datetime.date.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, 15)]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predictions": future_predictions
    })
    
    return df_copy, future_df




