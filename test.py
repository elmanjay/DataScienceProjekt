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

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.LUX])





app.layout = dbc.Container([
    dbc.Row([
    dbc.Navbar(
        [
            dbc.NavbarBrand("Mein Dashboard"),
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Über uns", href="#")),
        ],
        color="primary",
        dark=True,
    ),]),
    dbc.Row([
             dbc.Col(
    dbc.RadioItems(id="zeitraum", 
    options=[
        {'label': "Max", 'value': "max"},
        {'label': "Letzte 3 Monate", 'value': 3},
        {'label': "Letzte 6 Monate", 'value': 6}
    ],
    value="max",
    className="btn-group btn-group-sms",
    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
),)]),
    dcc.Graph(id="graph"),
    html.Table(id="table"),
    dcc.Dropdown(id="aktien-dropdown",
                  options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
                placeholder="Bitte wälen Sie eine Aktie"),
    dcc.Graph(id="graph2"),

    # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data")
],fluid=True)

@app.callback(Output("basic-data", "data"), Input("aktien-dropdown", "value"))
def clean_data(value):
     if value:
        msft = yf.Ticker(value)
        df = msft.history(period="max")
        df.reset_index(inplace= True)
        return df.to_json(date_format="iso", orient="split")

@app.callback(Output("graph", "figure"), Input("basic-data", "data"),Input("zeitraum","value"))
def update_graph(jsonified_cleaned_data, zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Kurs")
    return figure

@app.callback(Output("graph2", "figure"), Input("basic-data", "data"))
def decomposition_plot(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    decomposition = decompose(df)
    figure= px.line(decomposition, x="Date", y=["Trend", "Saison", "Rauschen"], title="Multiplikative Dekomposition")
    return figure

# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)