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
             dbc.Col([
    dbc.RadioItems(id="zeitraum", 
    options=[
        {'label': "Max", 'value': "max"},
        {'label': "Letzte 3 Monate", 'value': 3},
        {'label': "Letzte 6 Monate", 'value': 6}
    ],
    value="max",
    className="radiobuttons",
    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
    inline= True),
    dcc.Graph(id="graph")],
            width= 6),

        dbc.Col([
            html.Div( 
                children=[
                    html.H2("Marktdaten:", className="card-header"),
                    html.Div(
                        children=[
                            html.H3("Aktuelle Marktdaten"),
                            html.P("This is box 1 content")
                        ],
                        className="border-primary btn-close"
                    ),
                    html.Div(
                        children=[
                            html.H3("Historische Marktdaten"),
                            html.P("This is box 2 content")
                        ],
                        className="border-primary btn-close"
                    )
                ], 
                className="card border-primary mb-3"
            )
        ])
    ]),

    html.Table(id="table"),

    dcc.Graph(id="graph2"),

    dcc.Store(id="basic-data")
], fluid=True)

