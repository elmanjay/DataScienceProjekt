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

# Liste der Aktiensymbole und ihrer Namen
assets = ["ALV.DE", "AMZ.DE", "DPW.DE", "MDO.DE", "NVD.DE", "^MDAXI"]
aktien = ["Allianz", "Amazon", "Deutsche Post", "McDonald‘s", "NVIDIA", "MDAX"]

# Initialisierung der Dash-App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], use_pages=True)

# Layout der Dash-App
app.layout = dbc.Container([
    dbc.Row([
        # Navbar für die Navigation
        dbc.Navbar(
            [
                # Brand-Name der Navbar
                dbc.NavbarBrand("Aktien Analyse", className="navbar-brand", style={"margin-left": "10px"}),
                # NavLinks für die verschiedenen Seiten der Analyse
                dbc.NavItem(dbc.NavLink("Info", href="/")),
                dbc.NavItem(dbc.NavLink("Lineare Regression", href="/regression")),
                dbc.NavItem(dbc.NavLink("ARIMA", href="/zeitreihe")),
                dbc.NavItem(dbc.NavLink("LSTM", href="/lstm")),
                # Platzhalter für Abstand
                dbc.Col(
                    html.P(""),
                    width=2
                ),
                # Beschriftung für Dropdown-Auswahl
                dbc.Col(
                    html.P("Aktie auswählen:", style={"margin-right": "10px"}),
                    width=1
                ),
                # Dropdown für die Auswahl der Aktie
                dbc.Col(
                    dcc.Dropdown(
                        id="aktien-dropdown",
                        options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
                        value=assets[0],
                        style={"width": "300px", "color": "black"}
                    ),
                    width=True
                ),
            ],
            color="primary",
            dark=True,
            style={"color": "white"}
        ),
        # Alert für Fehlermeldungen
        dbc.Alert(
            "Bitte eine positive Ganzzahl eingeben!",
            id="alert-value-int",
            dismissable=True,
            is_open=False,
            className="alert alert-dismissible alert-danger"
        ),
        html.Hr(),
        # Container für die Seiteninhalte
        dbc.Container([
            dash.page_container,
            html.Br(),
            # Fußzeile
            html.Div(
                dbc.Col(html.P("Projektseminar Business Analytics SoSe 2023",
                                style={"text-align": "center", "margin-top": "10px"}, className="text-tertiary")),
                className="fixed-bottom text-white bg-primary")
        ], fluid=True),
        # Store-Komponente zum Speichern der Basisdaten
        dcc.Store(id="basic-data")
    ]),
], fluid=True)

# Callback zum Bereinigen der Daten und Speichern in einer Store-Komponente
@app.callback(Output("basic-data", "data"), Input("aktien-dropdown", "value"))
def clean_data(value):
    if value:
        # Daten von Yahoo Finance abrufen 
        msft = yf.Ticker(value)
        df = msft.history(period="max")
        df.reset_index(inplace=True)
        return df.to_json(date_format="iso", orient="split")


# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)