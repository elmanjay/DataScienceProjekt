import pandas as pd
import numpy as np
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
import locale
import datetime
import math
import pytz
import sys
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "backend")
sys.path.append(module_dir)
from backend_arima import predict_arima

#Setzen des aktuellen Datums
now = datetime.datetime.now()
locale.setlocale(locale.LC_TIME, 'de_DE')

dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            #Card die den Graphen beinhaltet
            html.Div([
                html.H2("ARIMA:", className="card-header"),
                html.Hr(),
    dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Container([dbc.Row([dbc.Label("Auswahl der Datengrundlage:", className="font-weight-bold")]),
                  dbc.Row([dbc.RadioItems(
                                            id="zeitraum-arima",
                                            options=[
                                                {"label": "1 Jahr", "value": 1},
                                                {"label": "2 Jahre (empfohlen)", "value": 2},
                                                {"label": "5 Jahre", "value": 5},
                                            ],
                                            value=2,
                                            className="radiobuttons",
                                            labelStyle={"display": "inline-block", "margin-right": "5px"},
                                            inline=True,
                                        )])                     
                                       
                                       ])
                        ),
                dbc.Col(dbc.Container([dbc.Row([dbc.Label("Auswahl der Paramenter:", className="font-weight-bold")]),
                  dbc.Row([dbc.Container([
                        dbc.Row(
            [
                dbc.Col([dbc.Input(type="text",inputMode="numeric",pattern="[0-9]*",placeholder="P=1",className="small-input",id="input-p")]),
                dbc.Col([dbc.Input(type="text",inputMode="numeric",pattern="[0-9]*",placeholder="D=2",className="small-input",id="input-d")]),
                dbc.Col([dbc.Input(type="text",inputMode="numeric",pattern="[0-9]*",placeholder="Q=1",className="small-input",id="input-q")]),
            ]
        )
    ]
)])                     
                                       
                                       ])
                        ),
            ]
        )
    ],
        fluid=True,
    ),

                html.Hr(),
                dcc.Graph(id="graph_arima")
            ], className="card text-white bg-primary mb-3", style={"height": "97.5%"})
        ], width=6),
        dbc.Col([
            dbc.Container([
                dbc.Row([
                    #Card mit den Performancemaßen
                    html.Div([
                        html.H2("Performance:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="output-div-performance-arima", style={"margin-left": "10px"})
                    ], className="card text-white bg-primary mb-3")
                ]),
                dbc.Row([
                    #Card mit den Prognosedaten
                    html.Div([
                        html.H2("Prognose:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="future-pred-table-arima", style={"margin-left": "10px"})
                    ], className="card border-primary mb-3")
                ])
            ])
        ], width=6)
    ]),
    #Store-Komponente zum Speichern der Basisdaten
    dcc.Store(id="basic-data"),
    #Store-Komponenten zum Speichern der ARIMA-Daten
    dcc.Store(id="prediction-arima"),
    dcc.Store(id="metrics-arima")
], fluid=True)

#Überprüfen ob Parameter ARIMA korrekt gewählt sind
@dash.callback(Output("alert-value-int","is_open") ,Input("input-p","value"),Input("input-d","value"),Input("input-q","value"))

def update_prediction_data(p_value, d_value, q_value):
    #Error Ausgabe bei falscher Wahl von p,d,q
    error= False
    if p_value and not p_value.isdigit():
        error = True
    elif d_value and not d_value.isdigit():
        error = True
    elif q_value and not q_value.isdigit():
        error = True

    return error

#Berechnung der Vorhersage und  Ausgabe der Ergebnisse
@dash.callback(Output("prediction-arima","data"),Output("metrics-arima","data"),Input("basic-data","data"),Input("zeitraum-arima","value"),Input("input-p","value"),Input("input-d","value"),Input("input-q","value"))

def update_prediction_data(basic_data,zeitraum,p_value, d_value, q_value):
    p_real= 1
    d_real = 2
    q_real = 1
    #Überprüfen ob Parameter richtig gewählt wurden
    if p_value and  p_value.isdigit() :
        p_real= int(p_value)
    if d_value and  d_value.isdigit() :
        d_real= int(d_value)
    if q_value and  q_value.isdigit() :
        q_real= int(q_value)
    #Vorhersage
    df = pd.read_json(basic_data, orient="split")
    prediction, metrics = predict_arima(df,p=p_real,d=d_real,q=q_real , years= zeitraum)

    return prediction.to_json(date_format="iso", orient="split"), json.dumps(metrics)

#Erstellen des Prognose-Graphen  
@dash.callback(Output("graph_arima","figure"),Input("prediction-arima","data"))

def update_prediction_data(prediction_data):
    df = pd.read_json(prediction_data, orient="split")
    df["Date"] = pd.Series(df["Date"], dtype="string")  
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})') 
    figure = px.scatter(template="plotly_dark")
    figure.add_trace(go.Scatter(x=df["Date"], y=df["Train"], mode="lines", name="Trainingsdaten"))
    figure.add_trace(go.Scatter(x=df["Date"], y=df["Test"], mode="lines", name="Testdaten"))
    figure.add_trace(go.Scatter(x=df["Date"], y= df["Prediction"], mode="lines", name="Vorhersage"))
    figure.update_layout(xaxis_title="Datum", yaxis_title="Kurs (EUR)", xaxis_type="category")
    figure.update_xaxes(tickformat="%Y-%m-%d")  
    
    # Anzahl der X-Achsenbeschriftungen festlegen
    num_ticks = 10

    # Manuelle Anpassung der X-Achsenbeschriftungen
    step = len(df["Date"]) // num_ticks
    tickvals = df["Date"][::step]
    figure.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals)

    return figure

#Erstellen Ausgabe der Prognosedaten
@dash.callback(Output("future-pred-table-arima", "children"), Input("prediction-arima", "data"), Input("basic-data", "data"))

def update_div_forecast(jsonified_cleaned_data, jsonified_cleaned_data_basic):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    df_basic= pd.read_json(jsonified_cleaned_data_basic, orient="split")
    df["Date"] = pd.Series(df["Date"], dtype="string") 
    df["Date"] = df["Date"].str.extract(r'^(\d{4}-\d{2}-\d{2})')
    today_value = df_basic["Close"].iloc[len(df_basic)-1]
    first_valid_index = df["Prediction"].first_valid_index()
    entwicklung_tomorrow = round((df["Prediction"].iloc[first_valid_index] - today_value) / today_value *100,2)
    entwicklung_week = round((df["Prediction"].iloc[first_valid_index+7] - today_value) / today_value *100,2)
    entwicklung_twoweek = round((df["Prediction"].iloc[first_valid_index+14] - today_value) / today_value *100, 2)
    entwicklungen = []

    for element in entwicklung_tomorrow, entwicklung_week, entwicklung_twoweek:
        if math.floor(element) >= 0:
            element = "+"+str(element) 
        entwicklungen.append(element)
    #Bestimmen der relevanten Datumseinträge
    today= datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    week = today + datetime.timedelta(days=7)
    twoweek=  today + datetime.timedelta(days=14)

    tomorrow_value = df[df["Date"] == str(tomorrow)]
    week_value = df[df["Date"] == str(week)]
    twoweek_value = df[df["Date"] == str(twoweek)]
    #Erstellen der Tabelle
    table_header = [
    html.Thead(html.Tr([html.Th(""), html.Th("Kursprognose"),html.Th("Entwicklung"), html.Th("Datum")]))]
    row1 = html.Tr([html.Td("Morgen"), html.Td(str(round(tomorrow_value["Prediction"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[0])+"%"), html.Td(tomorrow_value["Date"].iloc[0])])
    row2 = html.Tr([html.Td("7 Tage"), html.Td(str(round(week_value["Prediction"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[1])+"%"),html.Td(week_value["Date"].iloc[0])])
    row3 = html.Tr([html.Td("14 Tage"), html.Td(str(round(twoweek_value["Prediction"].iloc[0], 2))+"€"), html.Td(str(entwicklungen[2])+"%"),html.Td(twoweek_value["Date"].iloc[0])])
    table_body = [html.Tbody([row1, row2, row3])]
    table = dbc.Table(table_header + table_body, bordered=True, className="table-secondary table-hover card-body")
    
    return table

#Erstellen Ausgabe der Performancemaße
@dash.callback(Output("output-div-performance-arima", "children"), Input("metrics-arima","data"))

def update_div_performace(metrics_list):
    loaded_list = json.loads(metrics_list)
    #Aufrufen der Performancemaße
    mse= round(loaded_list[1],2)
    mae = round(loaded_list[0],2)
    smae = round(loaded_list[3]*100,2)
    rmse= round(loaded_list[2],2)
    #Erstellen Output-DIV
    output = [
        html.P("Mean Squared Error: {}".format(mse), className= "font-weight-bold"),
        html.P("Root Mean Square Error: {}".format(rmse), className= "font-weight-bold"),
        html.P("Mean Absolute Error: {}".format(mae), className= "font-weight-bold"),
        html.P("Scaled Mean Absolute Error: {}%".format(smae), className= "font-weight-bold")]

    return output
