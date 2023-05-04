import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend import decompose

assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft"]
dash.register_page(__name__)


layout = dbc.Container([
    dbc.Row([
             dbc.Col([
                 html.Div([
    html.H2("Dekomposition der Zeitreihe:", className= "card-header"),
    html.Label("Bitte wählen Sie die Art der Dekomposition:", style={"margin-left": "10px"}),
    dbc.RadioItems(id="radio-dekomposition", 
    options=[
        {'label': "Additiv", 'value': "add"},
        {'label': "Multiplikativ", 'value': "mult"},
    ],
    value="max",
    className="radiobuttons",
    labelStyle={'display': 'inline-block', 'margin-right': '5px'},
    style={"margin-left": "10px"},
    inline= True),
    html.Hr(),
    dcc.Graph(id="graph2")], className= "card border-primary mb-3")],
            ),])
                ])