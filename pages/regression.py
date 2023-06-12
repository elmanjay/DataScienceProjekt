import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend_regression import make_pred, make_pred_month
import plotly.graph_objects as go


assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft"]
dash.register_page(__name__)


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Lineare Regression:", className="card-header"),
                html.Hr(),
                html.Label("Bitte wählen Sie den gewünschten Zeitraum:",
                    style={"margin-left": "10px"}, className="font-weight-bold"),
                dbc.RadioItems(
                                id="zeitraum",
                                options=[
                                    {'label': "1 Monat (empfohlen)", 'value': 30},
                                    {'label': "3 Monate ", 'value': 90},
                                    {'label': "6 Monate", 'value': 180},
                                    {'label': "1 Jahr", 'value': 365}
                                ],
                                value=30,
                                className="radiobuttons",
                                labelStyle={'display': 'inline-block', 'margin-right': '5px'},
                                style={"margin-left": "10px"},
                                inline=True
                            ),
                html.Hr(),
                dcc.Graph(id="graph_regression")
            ], className="card text-white bg-primary mb-3", style={"height": "97.5%"})
        ], width=6),
        dbc.Col([
            dbc.Container([
                dbc.Row([
                    html.Div([
                        html.H2("Performance:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="output-div-performance", style={"margin-left": "10px"})
                    ], className="card text-white bg-primary mb-3")
                ]),
                dbc.Row([
                    html.Div([
                        html.H2("Prognose:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="future-pred-table", style={"margin-left": "10px"})
                    ], className="card border-primary mb-3")
                ])
            ])
        ], width=6)
    ]),
    dcc.Store(id="basic-data"),
    dcc.Store(id="regression-data"),
    dcc.Store(id="future-data")
], fluid=True)

@dash.callback(Output("regression-data", "data"),Output("future-data", "data"), Input("basic-data", "data"), Input("zeitraum","value") )

def generate_data(jsonified_cleaned_data,zeitraum):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    regression, futurregression = make_pred_month(df, zeitraum)
    regressiondata = regression.to_json(date_format="iso", orient="split")
    futuredata = futurregression.to_json(date_format="iso", orient="split")
    return regressiondata , futuredata


@dash.callback(Output("graph_regression", "figure"), Input("regression-data", "data"))

def update_graph(jsonified_cleaned_data):
    regression = pd.read_json(jsonified_cleaned_data, orient="split")
    figure= px.scatter(template= "plotly_dark")
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Train"], mode="markers", name="Trainingsdaten"))
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Test"], mode="markers", name="Testdaten"))
    figure.add_trace(go.Scatter(x=regression["Date"], y=regression["Predictions"], mode="lines", name="Vorhersage"))
    figure.update_layout(xaxis_title="Datum", yaxis_title="Kurs (USD)")
    figure.data[0].name = "Trainingsdaten"
    return figure 

@dash.callback(Output("output-div-performance", "children"), Input("regression-data", "data"))

def update_div_performace(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    r2= round(df["R2 Score"].iloc[0],2)
    mse= round(df["MSE"].iloc[0],2)
    mae = round(df["MAE"].iloc[0],2)
    smae = round(df["Scaled_mae"].iloc[0] * 100,2)
    rmse= round(df["RMSE"].iloc[0],2)
    output = [
        html.P("R2 Score: {}".format(r2), className= "font-weight-bold"),
        html.P("Mean Squared Error: {}".format(mse), className= "font-weight-bold"),
        html.P("Mean Absolute Error: {}".format(mae), className= "font-weight-bold"),
        html.P("Scaled Mean Absolute Error: {}%".format(smae), className= "font-weight-bold"),
        html.P("Root Mean Square Error: {}".format(rmse), className= "font-weight-bold"),
    ]
    return output

@dash.callback(Output("future-pred-table", "children"), Input("future-data", "data"), Input("basic-data", "data"))

def update_div_forecast(jsonified_cleaned_data, jsonified_cleaned_data_basic):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    df["Date"] = pd.Series(df["Date"], dtype="string")
    df_basic= pd.read_json(jsonified_cleaned_data_basic, orient="split")
    today_value = df_basic["Close"].iloc[len(df_basic)-1]
    entwicklung_tomorrow = round((df["Predictions"].iloc[0] - today_value) / today_value *100,2)
    entwicklung_week = round((df["Predictions"].iloc[6] - today_value) / today_value *100,2)
    entwicklung_twoweek = round((df["Predictions"].iloc[13] - today_value) / today_value *100, 2)
    entwicklungen = []
    for element in entwicklung_tomorrow, entwicklung_week, entwicklung_twoweek:
        if int(element) > 0:
            element = "+"+str(element) 
        entwicklungen.append(element)

    table_header = [
    html.Thead(html.Tr([html.Th(""), html.Th("Kursprognose"),html.Th("Entwicklung"), html.Th("Datum")]))]

    row1 = html.Tr([html.Td("Morgen"), html.Td(str(round(df["Predictions"].iloc[0], 2))+"$"), html.Td(str(entwicklungen[0])+"%"), html.Td(df["Date"].iloc[0])])
    row2 = html.Tr([html.Td("7 Tage"), html.Td(str(round(df["Predictions"].iloc[6], 2))+"$"), html.Td(str(entwicklungen[1])+"%"),html.Td(df["Date"].iloc[6])])
    row3 = html.Tr([html.Td("14 Tage"), html.Td(str(round(df["Predictions"].iloc[13], 2))+"$"), html.Td(str(entwicklungen[2])+"%"),html.Td(df["Date"].iloc[13])])

    table_body = [html.Tbody([row1, row2, row3])]

    table = dbc.Table(table_header + table_body, bordered=True, className="table-secondary table-hover card-body")
    return table 

@dash.callback(Output("regression-mainpage", "data"), Input("future-data", "data"))

def share_data(jsonified_cleaned_data):
    return jsonified_cleaned_data
