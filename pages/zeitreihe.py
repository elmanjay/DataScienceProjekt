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
    dcc.Graph(id="graph-trend"),
    dcc.Graph(id="graph-saison"),
    dcc.Graph(id="graph-rauschen")
    
    
    ], className= "card border-primary mb-3")
    
    ],
            ),]),
    dcc.Store(id="basic-data")
                ])

@dash.callback(Output("graph-trend", "figure"),Output("graph-saison", "figure"),Output("graph-rauschen", "figure"), Input("basic-data", "data"), Input("radio-dekomposition", "value"))
def decomposition_plot(jsonified_cleaned_data, radiodekomposition):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    df["Date"] = pd.to_datetime(df.Date).dt.tz_localize(None)
    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    zeitpunkt = now - datetime.timedelta(days=1825)
    zeitpunktformat = np.datetime64(zeitpunkt)
    df = df.loc[df["Date"] >= zeitpunktformat]
    decomposition = sm.tsa.seasonal_decompose(df["Close"], model= str(radiodekomposition), period=12)
    df["Trend"] = decomposition.trend
    df["Saison"] = decomposition.seasonal
    df["Rauschen"] = decomposition.resid
    figure1= px.line(df, x="Date", y=["Trend"],template= "plotly_white", title="Trend")
    figure1.update_xaxes(title_text="Datum")
    figure1.update_yaxes(title_text="Kurs (USD)")
    figure2= px.line(df, x="Date", y=["Saison"],template= "plotly_white", title="Saison")
    figure2.update_xaxes(title_text="Datum")
    figure2.update_yaxes(title_text="Kurs (USD)")
    figure3= px.line(df, x="Date", y=["Rauschen"],template= "plotly_white", title="Rauschen")
    figure3.update_xaxes(title_text="Datum")
    figure3.update_yaxes(title_text="Kurs (USD)")
    return figure1, figure2, figure3