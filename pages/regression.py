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
    html.H2("Lineare Regression:", className= "card-header"),
    dcc.Graph(id="graph_regression")], className= "card border-primary mb-3")],
            ),]),
    dcc.Store(id="basic-data")
                ])

@dash.callback(Output("graph_regression", "figure"), Input("basic-data", "data"))

def update_graph(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    regression = make_plot(df, 2019)
    figure= px.scatter(template= "plotly_white")
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Train"], mode="markers", name="Trainingsdaten"))
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Test"], mode="markers", name="Testdaten"))
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Predictions"], mode="lines", name="Vorhersage"))
    figure.update_layout(xaxis_title="Datum", yaxis_title="Kurs (USD)")
    figure.data[0].name = "Trainingsdaten"
    return figure