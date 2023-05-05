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

now = datetime.datetime.now()
locale.setlocale(locale.LC_TIME, 'de_DE')


dash.register_page(__name__, path='/')





layout = dbc.Container([
    dbc.Row([
             dbc.Col([
                 html.Div([
    html.H2("Verlauf der Aktie:", className= "card-header"),
    html.Label("Bitte wählen Sie den gewünschten Zeitraum:", style={"margin-left": "10px"}),
    dbc.RadioItems(id="zeitraum", 
    options=[
        {'label': "Max", 'value': "max"},
        {'label': "Letzte 3 Monate", 'value': 3},
        {'label': "Letzte 6 Monate", 'value': 6}
    ],
    value="max",
    className="radiobuttons",
    labelStyle={'display': 'inline-block', 'margin-right': '5px'},
    style={"margin-left": "10px"},
    inline= True),
    html.Hr(),
    dcc.Graph(id="graph")], className= "card border-primary mb-3")],
            width=6),

    dbc.Col([
        html.Div( 
    children= [
        html.H2("Aktuelle Marktdaten:", className= "card-header"),
        html.Hr(),
        html.H5(id="datumszeile-akt-markt"),
        html.Br(),
        html.Div(id="output-div-aktuellemarktdaten",style={"margin-left": "10px"}),
    ], className= "card text-white bg-primary mb-3")], width=3),
    dbc.Col([
        html.Div( 
    children= [
        html.H2("Historische Daten:", className= "card-header"),

    ], className= "card border-primary mb-3")], width=3)
]),
    dcc.Store(id="basic-data")
],fluid=True)

@dash.callback(Output("graph", "figure"), Input("basic-data", "data"),Input("zeitraum","value"))
def update_graph(jsonified_cleaned_data, zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie", template= "plotly_white")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs (USD)")
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
        html.P("Open: {}$".format(open_price)),
        html.P("Close: {}$".format(close_price)),
        html.P("Low: {}$".format(low_price)),
        html.P("High: {}$".format(high_price))
    ]
    return output

@dash.callback(Output("datumszeile-akt-markt", "children"),  Input("basic-data", "data"))

def update_datum(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    change_pct = (df["Open"].iloc[len(df)-1] - df["Open"].iloc[len(df)-2]) / df["Open"].iloc[len(df)-2] * 100
    change_pct = round(change_pct,2)

    if change_pct >0 :
        vorzeichen = "+"
        arrow= "▲"
    else: 
        vorzeichen = ""
        arrow= "▼"
    

    return html.H5(("{}{}{}%:  {}, {}".format(arrow, vorzeichen,change_pct, now.strftime("%A"), now.date())),style={"margin-left": "10px"}, className= "font-weight-bold")
