import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import datetime
import locale
import pytz
import numpy as np
from backend_regression import make_pred_reg
from backend_lstm import lstm_stock_prediction
from backend_arima import predict_arima

#Setzen des aktuellen Datums
now = datetime.datetime.now()
locale.setlocale(locale.LC_TIME, 'de_DE')

dash.register_page(__name__, path='/')

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    #Card die den Graphen beinhaltet
                    html.Div(
                        children=[
                            html.H2("Verlauf der Aktie:", className="card-header"),
                            html.Label("Bitte wählen Sie den gewünschten Zeitraum:",
                                       style={"margin-left": "10px"}, className="font-weight-bold"),
                            dbc.RadioItems(
                                id="zeitraum",
                                options=[
                                    {'label': "3 Monate", 'value': 90},
                                    {'label': "6 Monate", 'value': 180},
                                    {'label': "Max", 'value': "max"}
                                ],
                                value=90,
                                className="radiobuttons",
                                labelStyle={'display': 'inline-block', 'margin-right': '5px'},
                                style={"margin-left": "10px"},
                                inline=True
                            ),
                            html.Hr(style={"margin-top": "0px"}),
                            dcc.Graph(id="graph")
                        ],
                        className="card border-primary mb-3"
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Container(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        #Card mit den aktuellen Marktdaten
                                        html.Div(
                                            children=[
                                                html.H2("Aktuelle Marktdaten:", className="card-header"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.H5(id="datumszeile-akt-markt"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.Div(id="output-div-aktuellemarktdaten",
                                                         style={"margin-left": "10px"}),
                                            ],
                                            className="card text-white bg-primary mb-3 "
                                        ),
                                    width=6),
                                    dbc.Col(
                                        #Card mit den historischen Marktdaten
                                        html.Div(
                                            children=[
                                                html.H2("Historische Daten:", className="card-header"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.H5(id="datumszeile-hist-markt"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.Div(id="output-div-historischedaten",
                                                         style={"margin-left": "10px"}),
                                            ],
                                            className="card border-primary mb-3",
                                            style={"height": "96%"}
                                        ),
                                    width=6),
                                ],
                                className="mb-0 ",
                                align="stretch"
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        #Card mit zusammengefassten Prognosewerten
                                        html.Div(
                                            children=[
                                                html.H2("Prognose:", className="card-header"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.Div(id="prognose-div", style={"margin-left": "10px"}),
                                            ],
                                            className="card text-white bg-primary mb-3",
                                            style={"height": "115%"}
                                        ),
                                    ),
                                ],
                            className="my-0 " )
                        ]
                    ),
                    width=6
                )
            ]
        ),
    #Store-Komponente zum Speichern der Basisdaten
    dcc.Store(id="basic-data"),
    #Store-Komponenten zum Speichern der gefilterten Daten
    dcc.Store(id= "time-filtered-data")
],fluid=True)


# Callback für die Aktualisierung der gefilterten Daten basierend auf dem Zeitraum
@dash.callback(Output("time-filtered-data", "data"), Input("basic-data", "data"), Input("zeitraum", "value"))
def update_data(jsonified_cleaned_data, zeitraum):
    # Daten aus JSON laden
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    df["Date"] = pd.to_datetime(df.Date).dt.tz_localize(None)
    now = datetime.datetime.now(pytz.timezone("America/New_York"))

    # Daten basierend auf dem ausgewählten Zeitraum filtern
    if zeitraum != "max":
        zeitpunkt = now - datetime.timedelta(days=zeitraum)
        zeitpunktformat = np.datetime64(zeitpunkt)
        df = df.loc[df["Date"] >= zeitpunktformat]

    # Daten als JSON zurückgeben
    return df.to_json(date_format="iso", orient="split")


# Callback für die Aktualisierung des Graphen basierend auf den gefilterten Daten
@dash.callback(Output("graph", "figure"), Input("time-filtered-data", "data"))
def update_graph(jsonified_cleaned_data):
    # Daten aus JSON laden
    df = pd.read_json(jsonified_cleaned_data, orient="split")

    # Figure für den Plot erstellen
    figure = px.line(df, x="Date", y="Close", title="Verlauf der Aktie", template="plotly_white")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs (EUR)")

    # Figure zurückgeben
    return figure


# Callback für die Aktualisierung der aktuellen Marktdaten
@dash.callback(Output("output-div-aktuellemarktdaten", "children"), Input("aktien-dropdown", "value"))
def update_data(symbol):
    # Aktiendaten abrufen
    stock_data = yf.Ticker(symbol)
    data = stock_data.info
    open_price = data["regularMarketOpen"]
    close_price = data["regularMarketPreviousClose"]
    low_price = data["regularMarketDayLow"]
    high_price = data["regularMarketDayHigh"]

    # Output erstellen
    output = [
        html.P(f"Open: {open_price}€", className="font-weight-bold"),
        html.P(f"Previous Close: {close_price}€", className="font-weight-bold"),
        html.P(f"High: {high_price}€", className="font-weight-bold"),
        html.P(f"Low: {low_price}€", className="font-weight-bold"),
    ]

    return output


# Callback für die Aktualisierung der historischen Marktdaten
@dash.callback(Output("output-div-historischedaten", "children"), Input("time-filtered-data", "data"))
def update_data(jsonified_cleaned_data):
    # Daten aus JSON laden
    df = pd.read_json(jsonified_cleaned_data, orient="split")

    # Berechnungen durchführen
    all_time_high = round(df["Open"].max(), 2)
    all_time_low = round(df["Open"].min(), 2)
    all_time_mean = round(df["Open"].mean(), 2)
    all_time_std = round(df["Open"].std(), 2)

    # Output erstellen
    output = [
        html.P(f"High: {all_time_high}€", className="font-weight-bold"),
        html.P(f"Low: {all_time_low}€", className="font-weight-bold"),
        html.P(f"Average: {all_time_mean}€", className="font-weight-bold"),
        html.P(f"Standard Deviation: {all_time_std}", className="font-weight-bold")
    ]

    return output


# Callback für die Aktualisierung des Datumszeile der aktuellen Marktdaten
@dash.callback(Output("datumszeile-akt-markt", "children"), Input("aktien-dropdown", "value"))
def update_datum(symbol):
    # Aktiendaten abrufen
    stock_data = yf.Ticker(symbol)
    data = stock_data.info
    change_pct = (data["regularMarketOpen"] - data["regularMarketPreviousClose"]) / data["regularMarketPreviousClose"] * 100
    change_pct = round(change_pct, 2)

    # Vorzeichen und Pfeil basierend auf der berechneten Änderung festlegen
    if change_pct > 0:
        vorzeichen = "+"
        arrow = "▲"
    else:
        vorzeichen = ""
        arrow = "▼"

    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    output = html.H5(f"{arrow}{vorzeichen}{change_pct}%:  {now.strftime('%A')}, {now.date()}", style={"margin-left": "10px"}, className="font-weight-bold")

    return output


# Callback für die Aktualisierung des Datums der historischen Marktdaten
@dash.callback(Output("datumszeile-hist-markt", "children"), Input("time-filtered-data", "data"), Input("zeitraum", "value"))
def update_datum(jsonified_cleaned_data, zeitraum):
    # Daten aus JSON laden
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    change_pct = (df["Open"].iloc[len(df) - 1] - df["Open"].iloc[0]) / df["Open"].iloc[0] * 100
    change_pct = round(change_pct, 2)

    # Ausgabe-Zeitpunkt basierend auf dem Zeitraum festlegen
    if zeitraum == "max":
        ausgabe_zeitpunkt = "Seit Start:"
    elif zeitraum == 90:
        ausgabe_zeitpunkt = "3 Monate:"
    elif zeitraum == 180:
        ausgabe_zeitpunkt = "6 Monate:"

    # Vorzeichen basierend auf der Änderung berechnen
    if change_pct > 0:
        vorzeichen = "+"
    else:
        vorzeichen = ""

    output = html.H5(f"{ausgabe_zeitpunkt} {vorzeichen}{change_pct}%", style={"margin-left": "10px"}, className="font-weight-bold")

    return output


# Callback für die Aktualisierung der Prognosedaten
@dash.callback(Output("prognose-div", "children"), Input("aktien-dropdown", "value"), Input("basic-data", "data"))
def update_reg_main(symbol, data):
    # Aktiendaten abrufen
    stock_data = yf.Ticker(symbol)
    data_demand = stock_data.info
    close_price = data_demand["regularMarketPreviousClose"]
    forecasts = []
    percentage = []
    df = pd.read_json(data, orient="split")
    #Erstellen der Prognose
    result_regression = make_pred_reg(df, 30)
    result_lstm, metrics, futurelstm= lstm_stock_prediction(df, 365, ticker=symbol, prediction_days=14)
    result_arima , metrics = predict_arima(df,1,2,1)
    value_lstm = round(float(futurelstm["Predicted Future"].iloc[1]), 2)
    forecasts.append(round(result_regression[1]["Predictions"].iloc[0], 2))
    forecasts.append(round(value_lstm, 2))
    forecasts.append(round(result_arima["Prediction"].iloc[result_arima["Test"].last_valid_index()+2],2))

    #Festlegen Vorzeichen und Berecnung der Änderungsrate
    for element in forecasts:
        value = (element - close_price) / close_price * 100
        value = round(value, 2)
        if value > 0:
            value = "+" + str(value)
        percentage.append(value)

    #Erstellen des Output
    output = [
        html.P(f"Lineare Regression({percentage[0]}%): {forecasts[0]}€", className="font-weight-bold"),
        html.P(f"LSTM({percentage[1]}%): {forecasts[1]}€", className="font-weight-bold"),
        html.P(f"Arima({percentage[2]}%): {forecasts[2]}€", className="font-weight-bold")]

    return output
