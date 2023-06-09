import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "backend")
sys.path.append(module_dir)




assets = ["ALV.DE", "AMZ.DE", "DPW.DE", "MDO.DE", "NVD.DE","^MDAXI"]
aktien = ["Allianz","Amazon","Deutsche Post","McDonald‘s", "NVIDIA","MDAX"]

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.LUX],use_pages=True)





app.layout = dbc.Container([
    dbc.Row([
    dbc.Navbar(
        [
            dbc.NavbarBrand("Aktien Analyse", className= "navbar-brand",style={"margin-left": "10px"}),
            dbc.NavItem(dbc.NavLink("Info", href="/")),
            dbc.NavItem(dbc.NavLink("Lineare Regression", href="/regression")),
            dbc.NavItem(dbc.NavLink("ARIMA", href="/zeitreihe")),
            dbc.NavItem(dbc.NavLink("LSTM", href="/lstm")),
            dbc.Col(
                html.P(""),
                width=2  # Hier setzen wir die Breite der Beschriftungsspalte auf 2
            ),
            dbc.Col(
                html.P("Aktie auswählen:", style={"margin-right": "10px"}),
                width=1  # Hier setzen wir die Breite der Beschriftungsspalte auf 2
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="aktien-dropdown",
                    options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
                    value= assets[0],
                    style={"width": "300px","color": "black"}
                ),
                width=True   # Hier setzen wir die Breite der Dropdown-Spalte auf 6
            ),
        ],
        color="primary",
        dark=True,
        style={"color": "white"}
    ),
    dbc.Alert(
        "Bitte eine positive Ganzzahl eingeben!",
                                                                        id="alert-value-int",
                                                                        dismissable=True,
                                                                        is_open=False,
                                                                        className= "alert alert-dismissible alert-danger"
                                                                    ),
    html.Hr(),
    dbc.Container([
    dash.page_container,
    html.Br(),
    html.Div(
        dbc.Col(html.P("Projektseminar Business Analytics SoSe 2023", style={"text-align": "center","margin-top": "10px"},className="text-tertiary")),
        className="fixed-bottom text-white bg-primary")],fluid=True),
        # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data"),
    dcc.Store(id="regression-mainpage"),
    html.P(id="dummy")
    
]),
],fluid=True)

@app.callback(Output("basic-data", "data"), Input("aktien-dropdown", "value"))
def clean_data(value):
     if value:
        msft = yf.Ticker(value)
        df = msft.history(period="max")
        df.reset_index(inplace= True)
        return df.to_json(date_format="iso", orient="split")


# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)