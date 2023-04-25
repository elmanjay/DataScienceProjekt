import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Erstellen Sie ein Objekt für das zu beobachtende Unternehmen

assets = ["AAPL", "GOOGL", "TSLA", "MSFT", "ADS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft", "Adidas AG", "Bayer AG", "Bayerische Motoren Werke AG", "Daimler AG", "Bitcoin", "Ethereum", "Dogecoin", "XRP"]

# CSS-Stile
external_stylesheets = ['https://fonts.googleapis.com/css?family=Open+Sans&display=swap']

colors = {
    'background': '#0000FF',
    'text': '#7FDBFF'
}
dropdown_style = {
    'color': 'black',
    'backgroundColor': 'white',
    'border': '1px solid black',
    'width': "1008px",
    'fontFamily': "Open Sans",
    "font-weight" : "bold"
}
info_box_style = {
    'border': '1px solid black',
    'padding': '8px',
    'backgroundColor': 'white',
    'width': '300px',
    'fontFamily': 'Open Sans',
    "font-weight" : "bold"
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
        html.Label("Wählen Sie die zu untersuchende Aktie aus:", style={"padding": "10px 20px 10px 20px",'fontFamily': "Open Sans","text-decoration": "underline" } ),
        dcc.Dropdown(
            id='aktien-dropdown',
            options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],
            value=assets[0]
        )], style=dropdown_style),
        html.Hr(style={'border-top': '4px solid black'}),
        html.Div( children=[
        html.Label('Zeitraum:', style={'fontFamily': "Open Sans", "text-decoration": "underline","font-weight" : "bold" }),
        dcc.Dropdown(
        id='date-checklist',
        options=[
            {'label': '3 Monate', 'value': '3'},
            {'label': '6 Monate', 'value': '6'},
            {'label': 'Gesamter Verlauf', 'value': 'all'}
        ],value='3') ],style= {'border': '1px solid black',}),
        html.Div( children=[
            html.Div(
                children=[
                    dcc.Graph(id="timeline", style= graph)])
                    ,
            html.Div(
                children=[
                    html.Label('Informationen:', style={'fontFamily': "Open Sans", "text-decoration": "underline" }),
                    html.Ul([
                    dcc.Checklist(["Maximum", "Minimum", "Average"],
                    ["Maximum", "Minimum", "Average"],id= "checkbox",inline=True),
                    html.Hr(),
                    html.Li(id="max", style={}),
                    html.Li(id="min", style={}),
                    html.Li(id="average", style={})
                    ])
                        ],
                        style=info_box_style
                    )],style= {"display":"flex"},)
            ]
    )
#Anpassen des Plotts
@app.callback(
    Output(component_id='timeline', component_property='figure'),
    Input(component_id='aktien-dropdown', component_property='value'),
    Input(component_id="date-checklist", component_property='value' )
)
def update_output_div(input_value, date_range):
    msft = yf.Ticker(input_value)
    if date_range != "all":
        df= msft.history(period=str(date_range)+"mo")
    else: 
        df= msft.history(period="max")
    df.reset_index(inplace= True)
    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie")
    figure.update_xaxes(title_text="Datum")
    figure.update_yaxes(title_text="Eröffnungskurs")
    return figure


#Bearbeitung des Heads entsprechend der ausgewählten Aktie
@app.callback(
    Output(component_id='head', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    return "Untersuchung der {} Aktie".format(aktien[assets.index(input_value)])

#Checkbox ob Max Wert angezeibgt werden soll
@app.callback(
    Output(component_id='max', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value'),
    Input(component_id="checkbox", component_property="value")
)
def update_output_div(input_value,checkbox):
    if "Maximum" in checkbox:
        msft = yf.Ticker(input_value)
        df= msft.history(period="max")
        df.reset_index(inplace= True)  
        return 'All Time High: {} Dollar.'.format(round(df["Open"].max(),2 ))
    else:
        return ""
#Checkbox ob Min Wert angezeigt werden soll
@app.callback(
    Output(component_id='min', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value'),
    Input(component_id="checkbox", component_property="value")
)
def update_output_div(input_value,checkbox):
    if "Minimum" in checkbox:
        msft = yf.Ticker(input_value)
        df= msft.history(period="max")
        df.reset_index(inplace= True)  
        return 'All Time Low: {} Dollar.'.format(round(df["Open"].min(),2 ))
    else:
        return 
#Checkbox ob Average Wert angezeigt werden soll
@app.callback(
    Output(component_id='average', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value'),
    Input(component_id="checkbox", component_property="value")
)
def update_output_div(input_value,checkbox):
    if "Average" in checkbox:
        msft = yf.Ticker(input_value)
        df= msft.history(period="max")
        df.reset_index(inplace= True)  
        return 'Average: {} Dollar.'.format(round(df["Open"].mean(),2 ))
    else:
        return 
    
#Erstellen des Candle Chart
# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)