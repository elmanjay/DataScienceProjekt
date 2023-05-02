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

dash.register_page(__name__, path='/')





layout = dbc.Container([
    dbc.Row([
             dbc.Col(
    dbc.RadioItems(id="zeitraum", 
    options=[
        {'label': "Max", 'value': "max"},
        {'label': "Letzte 3 Monate", 'value': 3},
        {'label': "Letzte 6 Monate", 'value': 6}
    ],
    value="max",
    className="radiobuttons",
    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
),)]),
    dcc.Graph(id="graph"),
    html.Table(id="table"),
    #dcc.Dropdown(id="aktien-dropdown",
      #            options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
       #         placeholder="Bitte wälen Sie eine Aktie"),
    dcc.Graph(id="graph2"),

    # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data")
],fluid=True)
