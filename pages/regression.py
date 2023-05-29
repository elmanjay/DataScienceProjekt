import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from backend_regression import make_pred
import plotly.graph_objects as go


assets = ["AAPL", "GOOGL", "TSLA", "MSFT"]
aktien = ["Amazon", "Google", "Tesla", "Microsoft"]
dash.register_page(__name__)


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Lineare Regression:", className="card-header"),
                dcc.Graph(id="graph_regression")
            ], className="card border-primary mb-3")
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
                        html.H2("Vorhersage:", className="card-header"),
                        html.Hr(style={"margin-top": "0px"}),
                        html.Div(id="futre-pred-table", style={"margin-left": "10px"})
                    ], className="card text-white bg-primary mb-3")
                ])
            ])
        ], width=6)
    ]),
    dcc.Store(id="basic-data"),
    dcc.Store(id="regression-data"),
    dcc.Store(id="future-data")
], fluid=True)

@dash.callback(Output("regression-data", "data"),Output("future-data", "data"), Input("basic-data", "data"))

def generate_data(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient="split")
    regression, futurregression = make_pred(df, 2019)
    regressiondata = regression.to_json(date_format="iso", orient="split")
    futuredata = futurregression.to_json(date_format="iso", orient="split")
    return regressiondata , futuredata


@dash.callback(Output("graph_regression", "figure"), Input("regression-data", "data"))

def update_graph(jsonified_cleaned_data):
    regression = pd.read_json(jsonified_cleaned_data, orient="split")
    figure= px.scatter(template= "plotly_white")
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
    rmse= round(df["RMSE"].iloc[0],2)
    output = [
        html.P("R2 Score: {}".format(r2), className= "font-weight-bold"),
        html.P("Mean Squared Error: {}".format(mse), className= "font-weight-bold"),
        html.P("Mean Absolute Error: {}".format(mae), className= "font-weight-bold"),
        html.P("Root Mean Square Error: {}".format(rmse), className= "font-weight-bold"),
    ]
    return output

