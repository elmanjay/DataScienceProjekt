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
msft = yf.Ticker("MSFT")
df= msft.history(period="max")
df.reset_index(inplace= True)


# Erstelle das Dash-Layout
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        dcc.Dropdown(id='aktien-dropdown',options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],value=assets[0]),
        html.H1('Verlauf der Microsoft Aktie', style={'textAlign': 'center'}),
        dcc.Graph(id="timeline", figure=px.line(df, x="Date", y="Open"),style={'backgroundColor': '#000000'}),
        html.Br(),
        html.Div(id='my-output'),
])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='dropdown', component_property='value')
)
def update_output_div(input_value):
    return input_value

# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)
