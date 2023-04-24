import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Erstellen Sie ein Objekt für das zu beobachtende Unternehmen
assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon","Google","Tesla","Microsoft"]



colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
dropdown_style = {
    'color': 'black',
    'backgroundColor': 'white',
    'border': '1px solid black',
    'fontFamily': "Open Sans",
    "font-weight" : "bold"
}
info_box_style = {
    'border': '1px solid black',
    'padding': '8px',
    'backgroundColor': 'white',
    'fontFamily': 'Open Sans',
    "font-weight" : "bold",
}
graph = {
    'border': '1px solid black',
    'padding': '8px',
    'backgroundColor': 'white',
    'fontFamily': "Open Sans"  
}

# CSS-Stile
external_stylesheets = ['https://fonts.googleapis.com/css?family=Open+Sans&display=swap',
                        'https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css'
                        ]
# Bootstrap Design
external_scripts = ['https://code.jquery.com/jquery-1.12.4.min.js',
                    'https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js'
                    ]


# Erstelle das Dash-Layout
app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)
app.layout = html.Div(
    className="container",
    children=[
    html.Div(
            className="row",
            children=[
        html.H1('Text', id="head", style={'textAlign': 'center','fontFamily': 'Open Sans'}, className="col-md-12"),
        html.Div(
            className="col-md-10 col-md-offset-1",
            children=[
        html.Label("Wählen Sie die zu untersuchende Aktie aus:", style={'fontFamily': "Open Sans","text-decoration": "underline" } ),
        dcc.Dropdown(
            id='aktien-dropdown',
            options=[{'label': j, 'value': aktie} for j, aktie in zip(aktien, assets)],
            value=assets[0]
        )], style=dropdown_style)]),
         html.Div(
                className="row",
                children=[
        html.Hr( className="col-md-12", style={'border-top': '4px solid black'}),
            html.Div(
                className="col-md-7 col-md-offset-1",
                children=[
                    dcc.Graph(id="timeline",style= graph)])
                    ,
            html.Div(
                className="col-md-3",
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
                    )]#,style= {"display":"flex"}
                    )
                
                
    ]
    )
#Anpassen des Plotts
@app.callback(
    Output(component_id='timeline', component_property='figure'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    msft = yf.Ticker(input_value)
    df= msft.history(period="max")
    df.reset_index(inplace= True)
    figure= px.line(df, x="Date", y="Open", title="Verlauf der Aktie")
    return figure

#Bearbeitung des Heads entsprechend der ausgewählten Aktie
@app.callback(
    Output(component_id='head', component_property='children'),
    Input(component_id='aktien-dropdown', component_property='value')
)
def update_output_div(input_value):
    return "Untersuchung der {} Aktie".format(aktien[assets.index(input_value)])

#Bearbeitung der Info Box
#Checkbox ob Max Wert angezeigt werden soll
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
    

# Starte die App
if __name__ == '__main__':
    app.run_server(debug=True)
