import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend import decompose
import datetime
import locale
import pytz
import numpy as np
from backend_regression import make_pred_month

now = datetime.datetime.now()
locale.setlocale(locale.LC_TIME, 'de_DE')


dash.register_page(__name__, path='/')





layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
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
                                        html.Div(
                                            children=[
                                                html.H2("Prognose:", className="card-header"),
                                                html.Hr(style={"margin-top": "0px"}),
                                                html.Div(id="prognose-div", style={"margin-left": "10px"}),
                                            ],
                                            className="card text-white bg-primary mb-3",
                                            style={"height": "106%"}
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
    dcc.Store(id="basic-data"),
    dcc.Store(id= "time-filtered-data")
],fluid=True)

@dash.callback(Output("time-filtered-data", "data"), Input("basic-data", "data"),Input("zeitraum","value"))

def update_data(jsonified_cleaned_data, zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    df["Date"] = pd.to_datetime(df.Date).dt.tz_localize(None)
    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    if zeitraum != "max":
        zeitpunkt = now - datetime.timedelta(days=zeitraum)
        zeitpunktformat = np.datetime64(zeitpunkt)
        df = df.loc[df["Date"] >= zeitpunktformat]
    return df.to_json(date_format="iso", orient="split")

@dash.callback(Output("graph", "figure"), Input("time-filtered-data", "data"))

def update_graph(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    figure= px.line(df, x="Date", y="Close", title="Verlauf der Aktie", template= "plotly_white")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs (EUR)")
    return figure

@dash.callback(Output("output-div-aktuellemarktdaten", "children"), Input("aktien-dropdown", "value"))

def update_data(symbol):
    stock_data = yf.Ticker(symbol)
    data = stock_data.info
    open_price = data['regularMarketOpen']
    close_price = data['regularMarketPreviousClose']
    low_price = data['regularMarketDayLow']
    high_price = data['regularMarketDayHigh']
    output = [
        html.P("Open: {}€".format(open_price), className= "font-weight-bold"),
        html.P("Previous Close: {}€".format(close_price), className= "font-weight-bold"),
        html.P("High: {}€".format(high_price), className= "font-weight-bold"),
        html.P("Low: {}€".format(low_price), className= "font-weight-bold"),
    ]
    return output

@dash.callback(Output("output-div-historischedaten", "children"),Input("time-filtered-data", "data"))

def update_data(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')   
    all_time_high = round(df["Open"].max(), 2)
    all_time_low = round(df["Open"].min(), 2)
    all_time_mean = round(df["Open"].mean(), 2)
    all_time_std = round(df["Open"].std(), 2)
    output = [
        html.P("High: {}€".format(all_time_high), className= "font-weight-bold"),
        html.P("Low: {}€".format(all_time_low), className= "font-weight-bold"),
        html.P("Average: {}€".format(all_time_mean), className= "font-weight-bold"),
        html.P("Standard Deviation: {}".format(all_time_std), className="font-weight-bold")
    ]
    return output

@dash.callback(Output("datumszeile-akt-markt", "children"), Input("aktien-dropdown", "value") )

def update_datum(symbol):
    stock_data = yf.Ticker(symbol)
    data = stock_data.info
    change_pct = (data['regularMarketOpen'] - data['regularMarketPreviousClose']) / data['regularMarketPreviousClose'] * 100
    change_pct = round(change_pct,2)
   

    if change_pct >0 :
        vorzeichen = "+"
        arrow= "▲"
    else: 
        vorzeichen = ""
        arrow= "▼"
    

    return html.H5(("{}{}{}%:  {}, {}".format(arrow, vorzeichen,change_pct, now.strftime("%A"), now.date())),style={"margin-left": "10px"}, className= "font-weight-bold")

@dash.callback(Output("datumszeile-hist-markt", "children"),  Input("time-filtered-data", "data"),Input("zeitraum","value"))

def update_datum(jsonified_cleaned_data,zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    change_pct = (df["Open"].iloc[len(df)-1] - df["Open"].iloc[0]) / df["Open"].iloc[0] * 100
    change_pct = round(change_pct,2)

    if zeitraum == "max":
        ausgabe_zeitpunkt = "Seit Start:"
    elif zeitraum == 90:
        ausgabe_zeitpunkt = "3 Monate:"
    elif zeitraum == 180:
        ausgabe_zeitpunkt = "6 Monate:"
    if change_pct >0 :
        vorzeichen = "+"    
    else: 
        vorzeichen = ""
    return html.H5(("{} {}{}%".format(ausgabe_zeitpunkt, vorzeichen,change_pct, )),style={"margin-left": "10px"}, className= "font-weight-bold")

@dash.callback(Output("prognose-div", "children"),Input("aktien-dropdown", "value"), Input("basic-data", "data"))
def update_reg_main(symbol, data):
    stock_data = yf.Ticker(symbol)
    data_demand = stock_data.info
    close_price = data_demand ['regularMarketPreviousClose']
    forecasts = []
    percentage = []
    vorzeichen_liste = []
    df = pd.read_json(data, orient="split")
    result_regression= make_pred_month(df, 30)
    forecasts.append(round(result_regression[1]["Predictions"].iloc[0],2))

    for element in forecasts:
        value = (element - close_price) / close_price *100
        value= round(value,2)
        if value >0 :
            value = "+" + str(value)
        percentage.append(value)
    


    output = [
        html.P("Lineare Regression({}%): {}€".format(percentage[0],forecasts[0]), className= "font-weight-bold"),
        html.P("Arima: {}€".format(forecasts[0]), className= "font-weight-bold"),
        html.P("LSTM: {}€".format(forecasts[0]), className= "font-weight-bold")
    ]
    return output