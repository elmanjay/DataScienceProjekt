import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend_regression import make_plot
import plotly.graph_objects as go

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
    dcc.Graph(id="graph_regression")], className= "card border-primary mb-3")],
            ),]),
    dcc.Store(id="basic-data")
                ])

@dash.callback(Output("graph_regression", "figure"), Input("basic-data", "data"))

def update_graph(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    regression = make_plot(df)
    figure= px.line(regression , x="Date", y="Close", title="Verlauf der Aktie", template= "plotly_white")
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Predictions"], mode="lines", name="Regression"))

    return figure