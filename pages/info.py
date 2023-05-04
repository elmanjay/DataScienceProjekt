import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend import decompose



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
            ),

    dbc.Col([
        html.Div( 
    children= [
        html.H2("Aktuelle Marktdaten:", className= "card-header"),
        html.Content("test eijfjfosjfsjdojesdjsjefosjdlsejflsej",className= "card-text"),
        html.P("test")
    ], className= "card border-primary mb-3")])
]),

    html.Table(id="table"),
    #dcc.Dropdown(id="aktien-dropdown",
      #            options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
       #         placeholder="Bitte wälen Sie eine Aktie"),
    dcc.Graph(id="graph2"),

    # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data")
],fluid=True)

@dash.callback(Output("graph", "figure"), Input("basic-data", "data"),Input("zeitraum","value"))
def update_graph(jsonified_cleaned_data, zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie", template= "plotly_white")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs (USD)")
    return figure
