import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Erstellen Sie ein Objekt für das zu beobachtende Unternehmen
assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon","Googel","Tesla","Microsoft"]


# Erstelle das Dash-Layout
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1('Text', id="head", style={'textAlign': 'center'}),
        dcc.Dropdown(id='aktien-dropdown',options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],value=assets[0]),
        dcc.Graph(id="timeline")
        ])

@app.callback(
    Output(component_id='timeline', component_property='figure'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    msft = yf.Ticker(input_value)
    df= msft.history(period="max")
    df.reset_index(inplace= True)
    figure= px.line(df, x="Date", y="Open")
    return figure

#Bearbeitung des Heads
@app.callback(
    Output(component_id='head', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    return 'Verlauf der {} Aktie'.format(input_value)
    

# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)
