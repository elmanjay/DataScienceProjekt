import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from backend import decompose



assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft"]

app = dash.Dash(__name__)

external_stylesheets = [    {        'external_url': 'https://fonts.googleapis.com/css?family=SF+Pro+Display:400,600,700'    }]





app.layout = html.Div([
    html.H1('Mein Dashboard', style={'font-family': 'SF Pro Display'}),
    #dcc.RadioItems( options=[
       # {'label': "Max", "value": 3},
       # {'label': "Letzte 3 Monate", "value": 6},
       # {'label': "Letzte 6 Monate", "value": len(df["Date"].unique())}],
          #"Max", id= "zeitraum",inline= True),
    dcc.Graph(id="graph"),
    html.Table(id="table"),
    dcc.Dropdown(id="aktien-dropdown", options=[{"label": j, "value": aktie} for j, aktie in zip(aktien, assets)],
            value=assets[0]),
    dcc.Graph(id="graph2"),

    # dcc.Store stores the intermediate value
    dcc.Store(id="basic-data")
])

@app.callback(Output("basic-data", "data"), Input("aktien-dropdown", "value"))
def clean_data(value):
     msft = yf.Ticker(value)
     df = msft.history(period="max")
     df.reset_index(inplace= True)

     return df.to_json(date_format="iso", orient="split")

@app.callback(Output("graph", "figure"), Input("basic-data", "data"))
def update_graph(jsonified_cleaned_data):

    df = pd.read_json(jsonified_cleaned_data, orient='split')

    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Eröffnungskurs")
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