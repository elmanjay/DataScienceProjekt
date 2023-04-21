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

# CSS-Stile
external_stylesheets = ['https://fonts.googleapis.com/css?family=Open+Sans&display=swap']

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
dropdown_style = {
    'color': 'black',
    'backgroundColor': 'white',
    'border': '1px solid black',
    'width': '800px',
    'fontFamily': "Open Sans"
}
info_box_style = {
    'border': '1px solid black',
    'padding': '8px',
    'backgroundColor': 'white',
    'width': '300px',
    'fontFamily': 'Open Sans'
}
graph = {
    'border': '1px solid black',
    'padding': '8px',
    'backgroundColor': 'white',
    'width': '800px',
    'fontFamily': "Open Sans"  
}


# Erstelle das Dash-Layout
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1('Text', id="head", style={'textAlign': 'center','fontFamily': 'Open Sans'}),
        html.Div(
            children=[
        html.H2("Wählen Sie die zu untersuchende Aktie aus:"),
        dcc.Dropdown(
            id='aktien-dropdown',
            options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],
            value=assets[0]
        )], style=dropdown_style),
        html.Div(
            children=[
                dcc.Graph(id="timeline", style= graph)
                ,
        html.Div(
            children=[
                html.H2('Informationen', style={}),
                html.Ul([
                html.Li(id="max", style={}),
                html.Li(id="min", style={}),])
                    ],
                    style=info_box_style
                )
                ])
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
