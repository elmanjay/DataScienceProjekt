import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
import plotly.express as px

# Erstellen Sie ein Objekt für das zu beobachtende Unternehmen
msft = yf.Ticker("MSFT")
df= msft.history(period="max")
df.reset_index(inplace= True)


# Erstelle das Dash-Layout
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1('Verlauf der Microsoft Aktie', style={'textAlign': 'center'}),
        dcc.Graph(id='example-graph', figure=px.line(df, x="Date", y="Open"),style={'backgroundColor': '#000000'}),
        dcc.Dropdown(options=[{'label': 'Microsoft', 'value': 'option1'},{"label": 'Netflix', "value": 'option2'},
        {'label': 'Tesla', 'value': 'option3'}])
])


# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)
