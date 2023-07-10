import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "backend")
sys.path.append(module_dir)
from backend_lstm import lstm_stock_prediction
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
                html.Label("Visualisierung von Trainingsdaten, Testdaten und Vorhersagen:",
                    style={"margin-left": "10px"}, className="font-weight-bold"),
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
                        html.Div(id="output-div-performance-lstm", style={"margin-left": "10px"})
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
    dcc.Store(id="basic-data"),
    dcc.Store(id="prediction_lstm"),
    dcc.Store(id="metrics_lstm"),
    dcc.Store(id="future_lstm")
], fluid=True)


@dash.callback(Output("prediction_lstm", "data"),Output("metrics_lstm", "data"),Output("future_lstm", "data"), Input("basic-data", "data"), Input("aktien-dropdown", "value"))

def save_data_lstm(json_data,ticker):
    df = pd.read_json(json_data, orient="split")
    prediction_full,metrics, futureprediction   = lstm_stock_prediction(df, 1080,ticker, prediction_days=14)
    return prediction_full.to_json(date_format="iso", orient="split"), metrics.to_json(date_format="iso", orient="split"), futureprediction.to_json(date_format="iso", orient="split")



     
@dash.callback(Output("graph_lstm", "figure"),Input("prediction_lstm", "data"))

def update_graph_lstm(prediction):
    prediction_data = pd.read_json(prediction, orient="split")
    prediction_data.reset_index(inplace=True,names="Date")
    prediction_data["Date"] = pd.Series(prediction_data["Date"], dtype="string") 
    prediction_data["Date"] = prediction_data["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    figure = px.scatter(template="plotly_dark")
    figure.add_trace(go.Scatter(x=prediction_data["Date"], y=prediction_data["Train"], mode="lines", name="Trainingsdaten"))
    figure.add_trace(go.Scatter(x=prediction_data["Date"], y=prediction_data["Test"], mode="lines", name="Testdaten"))
    figure.add_trace(go.Scatter(x=prediction_data["Date"], y=prediction_data["Predicted Test"], mode="lines", name="Vorhersage Testdaten"))
    figure.add_trace(go.Scatter(x=prediction_data["Date"], y= prediction_data["Predicted Future"], mode="lines", name="Vorhersage Zukunft"))
    figure.update_layout(xaxis_title="Datum", yaxis_title="Kurs (EUR)", xaxis_type="category")
    figure.update_xaxes(tickformat="%Y-%m-%d")  # X-Achsenbeschriftung im gewünschten Format festlegen
    
    # Anzahl der X-Achsenbeschriftungen festlegen
    num_ticks = 10

    # Werte und Beschriftungen für die X-Achsenbeschriftung auswählen
    step = len(prediction_data["Date"]) // num_ticks
    tickvals = prediction_data["Date"][::step]
    #ticktext = [date.strftime("%Y-%m-%d") for date in tickvals]

    # Manuelle Anpassung der X-Achsenbeschriftungen
    figure.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals
    )

    #figure.data[0].name = "Trainingsdaten"
    return figure

@dash.callback(Output("future-pred-table-lstm", "children"), Input("future_lstm", "data"), Input("basic-data", "data"))

def update_div_forecast(jsonified_cleaned_data, jsonified_cleaned_data_basic):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    df_basic= pd.read_json(jsonified_cleaned_data_basic, orient="split")
    df.reset_index(inplace=True,names="Date")
    df["Date"] = pd.Series(df["Date"], dtype="string") 
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    today_value = df_basic["Close"].iloc[len(df_basic)-1]
    entwicklung_tomorrow = round((df["Predicted Future"].iloc[0] - today_value) / today_value *100,2)
    entwicklung_week = round((df["Predicted Future"].iloc[6] - today_value) / today_value *100,2)
    entwicklung_twoweek = round((df["Predicted Future"].iloc[13] - today_value) / today_value *100, 2)
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

    row1 = html.Tr([html.Td("Morgen"), html.Td(str(round(tomorrow_value["Predicted Future"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[0])+"%"), html.Td(tomorrow_value["Date"].iloc[0])])
    row2 = html.Tr([html.Td("7 Tage"), html.Td(str(round(week_value["Predicted Future"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[1])+"%"),html.Td(week_value["Date"].iloc[0])])
    row3 = html.Tr([html.Td("14 Tage"), html.Td(str(round(twoweek_value["Predicted Future"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[2])+"%"),html.Td(twoweek_value["Date"].iloc[0])])

    table_body = [html.Tbody([row1, row2, row3])]

    table = dbc.Table(table_header + table_body, bordered=True, className="table-secondary table-hover card-body")
    return table


@dash.callback(Output("output-div-performance-lstm", "children"), Input("metrics_lstm", "data"))

def update_div_performace(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    mse= round(df["MSE"].iloc[0],2)
    mae = round(df["MAE"].iloc[0],2)
    smae = round(df["Scaled MAE"].iloc[0] * 100,2)
    rmse= round(df["RMSE"].iloc[0],2)
    output = [
        html.P("Mean Squared Error: {}".format(mse), className= "font-weight-bold"),
        html.P("Root Mean Square Error: {}".format(rmse), className= "font-weight-bold"),
        html.P("Mean Absolute Error: {}".format(mae), className= "font-weight-bold"),
        html.P("Scaled Mean Absolute Error: {}%".format(smae), className= "font-weight-bold"),
    ]
    return output

