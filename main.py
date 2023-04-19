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
        dcc.Dropdown(
            id='aktien-dropdown',
            options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],
            value=assets[0]
        ),
        html.Div(
            children=[
                dcc.Graph(id="timeline"),
                html.Div(
                    children=[
                        html.H2('Informationen'),
                        html.Ul([
                            html.Li(id="max"),
                            html.Li(id="min"),
                        ])
                    ],
                    style={'border': '2px solid #ccc', 'padding': '8px'}
                )
            ],
            style={'width': '70%', 'display': 'inline-block'}
        )
    ]
)

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

#Bearbeitung der Info BOx
@app.callback(
    Output(component_id='max', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    msft = yf.Ticker(input_value)
    df= msft.history(period="max")
    df.reset_index(inplace= True)  

    return 'All Time High: {} Dollar.'.format(round(df["Open"].max(),2 ))

@app.callback(
    Output(component_id='min', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    msft = yf.Ticker(input_value)
    df= msft.history(period="max")
    df.reset_index(inplace= True)  

    return 'All Time Low: {} Dollar.'.format(round(df["Open"].min(),2 ))
    

# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)
