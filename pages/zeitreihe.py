import pandas as pd
import numpy as np
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.api as sm
import locale
import datetime
import pytz

now = datetime.datetime.now()
locale.setlocale(locale.LC_TIME, 'de_DE')

dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("ARIMA:", className="card-header"),
                html.Hr(),
                html.Label("Bitte wählen Sie den gewünschten Zeitraum:",
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
    dcc.Store(id="basic-data"),
    dcc.Store(id="prediction"),
    dcc.Store(id="metrics")
], fluid=True)

