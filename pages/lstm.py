import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend_lstm import give_results2
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np



assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft"]

dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("LSTM:", className="card-header"),
                html.Hr(),
                html.Label("Bitte wählen Sie den gewünschten Zeitraum:",
                    style={"margin-left": "10px"}, className="font-weight-bold"),
                dbc.RadioItems(
                                id="zeitraum",
                                options=[
                                    {'label': "1 Monat (empfohlen)", 'value': 30},
                                    {'label': "3 Monate ", 'value': 90},
                                    {'label': "6 Monate", 'value': 180},
                                    {'label': "1 Jahr", 'value': 365}
                                ],
                                value=30,
                                className="radiobuttons",
                                labelStyle={'display': 'inline-block', 'margin-right': '5px'},
                                style={"margin-left": "10px"},
                                inline=True
                            ),
                html.Hr(),
                dcc.Graph(id="graph_lstm")
            ], className="card text-white bg-primary mb-3", style={"height": "97.5%"})
        ], width=6),
        dbc.Col([
            dbc.Container([
                dbc.Row([
                    html.Div([
                        html.H2("Performance:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="output-div-performance", style={"margin-left": "10px"})
                    ], className="card text-white bg-primary mb-3")
                ]),
                dbc.Row([
                    html.Div([
                        html.H2("Prognose:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="future-pred-table-lstm", style={"margin-left": "10px"})
                    ], className="card border-primary mb-3")
                ])
            ])
        ], width=6)
    ]),
    dcc.Store(id="test-lstm"),
    dcc.Store(id="train-lstm"),
    dcc.Store(id="prediction"),
    dcc.Store(id="metrics")
], fluid=True)


@dash.callback(Output("train-lstm", "data"),Output("test-lstm", "data"),Output("prediction", "data"), Input("basic-data", "data"), Input("aktien-dropdown", "value"))

def save_data_lstm(json_data,ticker):
    df = pd.read_json(json_data, orient="split")
    train, test, prediction  = give_results2(df, 365,ticker, prediction_days=14)

    return train.to_json(date_format="iso", orient="split"), test.to_json(date_format="iso", orient="split"),prediction.to_json(date_format="iso", orient="split")



     
@dash.callback(Output("graph_lstm", "figure"),Input("basic-data","data"),Input("prediction", "data"))

def update_graph_lstm(basicdata,prediction):
    df = pd.read_json(basicdata, orient="split") # Erstelle eine Kopie des DataFrames, um Änderungen daran vorzunehmen
    df = df.drop(["Open", "High","Low","Volume","Dividends","Stock Splits"], axis=1)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)  # Konvertiere das Datum in das richtige Format
    now = datetime.datetime.now(pytz.timezone('America/New_York'))  # Aktuelles Datum und Uhrzeit in der Zeitzone New York
    zeitpunkt = now - datetime.timedelta(days=365)  # Berechne den Zeitpunkt basierend auf der angegebenen Anzahl von Tagen
    zeitpunktformat = np.datetime64(zeitpunkt)  # Konvertiere den Zeitpunkt in das richtige Format
    df = df.loc[df["Date"] >= zeitpunktformat]  # Filtere den DataFrame nach dem Zeitpunkt
    df["Date"] = pd.Series(df["Date"], dtype="string")  # Konvertiere das Datum in einen String
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')  # Extrahiere das Datum im Format 'YYYY-MM-DD
    train_len = int(len(df) * 0.92)
    train_data = df[:train_len]
    test_data = df[train_len - 2:]
    prediction_data = pd.read_json(prediction, orient="split")
    figure = px.scatter(template="plotly_dark")
    figure.add_trace(go.Scatter(x=train_data["Date"], y=train_data["Close"], mode="lines", name="Trainingsdaten"))
    figure.add_trace(go.Scatter(x=test_data["Date"], y=test_data["Close"], mode="lines", name="Testdaten"))
    figure.add_trace(go.Scatter(x=prediction_data.index, y= prediction_data["Predicted Close"], mode="lines", name="Vorhersage"))
    figure.update_layout(xaxis_title="Datum", yaxis_title="Kurs (EUR)", xaxis_type="category")
    figure.update_xaxes(tickformat="%Y-%m-%d")  # X-Achsenbeschriftung im gewünschten Format festlegen
    
    # Anzahl der X-Achsenbeschriftungen festlegen
    num_ticks = 10

    # Werte und Beschriftungen für die X-Achsenbeschriftung auswählen
    step = len(df["Date"]) // num_ticks
    tickvals = df["Date"][::step]
    #ticktext = [date.strftime("%Y-%m-%d") for date in tickvals]

    # Manuelle Anpassung der X-Achsenbeschriftungen
    figure.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals
    )

    #figure.data[0].name = "Trainingsdaten"
    return figure

@dash.callback(Output("future-pred-table-lstm", "children"), Input("prediction", "data"), Input("basic-data", "data"))

def update_div_forecast(jsonified_cleaned_data, jsonified_cleaned_data_basic):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    df_basic= pd.read_json(jsonified_cleaned_data_basic, orient="split")
    df.reset_index(inplace=True,names="Date")
    df["Date"] = pd.Series(df["Date"], dtype="string") 
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    print(df_basic.iloc[len(df_basic)-1])
    today_value = df_basic["Close"].iloc[len(df_basic)-1]
    entwicklung_tomorrow = round((df["Predicted Close"].iloc[0] - today_value) / today_value *100,2)
    entwicklung_week = round((df["Predicted Close"].iloc[6] - today_value) / today_value *100,2)
    entwicklung_twoweek = round((df["Predicted Close"].iloc[13] - today_value) / today_value *100, 2)
    entwicklungen = []
    for element in entwicklung_tomorrow, entwicklung_week, entwicklung_twoweek:
        if int(element) > 0:
            element = "+"+str(element) 
        entwicklungen.append(element)
    
    today= datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    week = today + datetime.timedelta(days=7)
    twoweek=  today + datetime.timedelta(days=14)

    tomorrow_value = df[df["Date"] == str(tomorrow) ]
    week_value = df[df["Date"] == str(week) ]
    twoweek_value = df[df["Date"] == str(twoweek) ]
    table_header = [
    html.Thead(html.Tr([html.Th(""), html.Th("Kursprognose"),html.Th("Entwicklung"), html.Th("Datum")]))]

    row1 = html.Tr([html.Td("Morgen"), html.Td(str(round(tomorrow_value["Predicted Close"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[0])+"%"), html.Td(tomorrow_value["Date"].iloc[0])])
    row2 = html.Tr([html.Td("7 Tage"), html.Td(str(round(week_value["Predicted Close"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[1])+"%"),html.Td(week_value["Date"].iloc[0])])
    row3 = html.Tr([html.Td("14 Tage"), html.Td(str(round(twoweek_value["Predicted Close"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[2])+"%"),html.Td(twoweek_value["Date"].iloc[0])])

    table_body = [html.Tbody([row1, row2, row3])]

    table = dbc.Table(table_header + table_body, bordered=True, className="table-secondary table-hover card-body")
    return table


