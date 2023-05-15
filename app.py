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

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.LUX],use_pages=True)





app.layout = dbc.Container([
    dbc.Row([
    dbc.Navbar(
        [
            dbc.NavbarBrand("Aktien Analyse", className= "navbar-brand",style={"margin-left": "10px"}),
            dbc.NavItem(dbc.NavLink("Info", href="/")),
            dbc.NavItem(dbc.NavLink("Regression", href="/lstm")),
            dbc.NavItem(dbc.NavLink("Zeitreihenanalyse", href="/zeitreihe")),
            dbc.NavItem(dbc.NavLink("LSTM", href="#")),
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
                    placeholder="Bitte wählen Sie eine Aktie",
                    style={"width": "300px","color": "black"}
                ),
                width=True   # Hier setzen wir die Breite der Dropdown-Spalte auf 6
            ),
        ],
        color="primary",
        dark=True,
        style={"color": "white"}
    ),
    html.Hr(),
    dash.page_container,
        # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data")
    
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