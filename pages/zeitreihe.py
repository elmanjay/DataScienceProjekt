import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.api as sm
from backend import decompose

dash.register_page(__name__)


layout = dbc.Container([
    dbc.Row([
             dbc.Col([
                 html.Div([
    html.H2("Dekomposition der Zeitreihe:", className= "card-header"),
    html.Label("Bitte wählen Sie die Parameter der Dekomposition:", style={"margin-left": "10px"}),
    dbc.RadioItems(id="radio-dekomposition", 
    options=[
        {'label': "Additiv", 'value': "additive"},
        {'label': "Multiplikativ", 'value': "multiplikative"},
    ],
    value="additive",
    className="radiobuttons",
    labelStyle={'display': 'inline-block', 'margin-right': '5px'},
    style={"margin-left": "10px"},
    inline= True),
    html.Hr(),
    dcc.Graph(id="graph2")], className= "card border-primary mb-3")],
            ),]),
    dcc.Store(id="basic-data")
                ])

@dash.callback(Output("graph2", "figure"), Input("basic-data", "data"), Input("radio-dekomposition", "value"))
def decomposition_plot(jsonified_cleaned_data, radiodekomposition):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    decomposition = sm.tsa.seasonal_decompose(df["Close"], model= str(radiodekomposition), period=12)
    df["Trend"] = decomposition.trend
    df["Saison"] = decomposition.seasonal
    df["Rauschen"] = decomposition.resid
    figure= px.line(df, x="Date", y=["Trend", "Saison", "Rauschen"],template= "plotly_white", title="Dekomposition")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs (USD)")
    return figure