import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go
import plotly.express as px

app = dash.Dash()

# Aktuelle Marktdaten
market_data = html.Div([
    html.H2('Aktuelle Marktdaten'),
    dash_table.DataTable(
    id='market-data-table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
    style_table={'overflowX': 'scroll'},
    style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                        )
    # hier können Sie Diagramme für die aktuellen Marktdaten hinzufügen
])

# Aktienliste
stock_list = html.Div([
    html.H2('Aktienliste'),
    # hier können Sie eine Tabelle mit ausgewählten Aktien hinzufügen
])

# Performance-Diagramme
performance_charts = html.Div([
    html.H2('Performance-Diagramme'),
    dcc.Graph(
        id='performance-chart',
        figure={
            'data': [
                go.Scatter(
                    x=[1, 2, 3],
                    y=[4, 1, 2],
                    mode='lines',
                    name='Aktie 1'
                ),
                go.Scatter(
                    x=[1, 2, 3],
                    y=[2, 4, 3],
                    mode='lines',
                    name='Aktie 2'
                )
            ],
            'layout': go.Layout(
                title='Aktienperformance',
                xaxis={'title': 'Zeit'},
                yaxis={'title': 'Preis'}
            )
        }
    )
])

# Vorhersage-Graphen
prediction_charts = html.Div([
    html.H2('Vorhersage-Graphen'),
    dcc.Graph(
        id='prediction-chart',
        figure={
            'data': [
                go.Scatter(
                    x=[1, 2, 3],
                    y=[4, 1, 2],
                    mode='lines',
                    name='Aktie 1'
                ),
                go.Scatter(
                    x=[1, 2, 3],
                    y=[2, 4, 3],
                    mode='lines',
                    name='Aktie 2'
                )
            ],
            'layout': go.Layout(
                title='Aktienvorhersage',
                xaxis={'title': 'Zeit'},
                yaxis={'title': 'Preis'}
            )
        }
    )
])

# Algorithmen
algorithms = html.Div([
    html.H2('Algorithmen'),
    # hier können Sie Algorithmen für Aktienvorhersagen hinzufügen
])

# Risikoanalyse
risk_analysis = html.Div([
    html.H2('Risikoanalyse'),
    # hier können Sie Diagramme oder Grafiken für die Risikoanalyse hinzufügen
])

# Benachrichtigungen
notifications = html.Div([
    html.H2('Benachrichtigungen'),
    # hier können Sie Benachrichtigungen hinzufügen, um den Benutzer über Änderungen in der Leistung von ausgewählten Aktien und Vorhersageergebnissen zu informieren
])

# Interaktive Tools
interactive_tools = html.Div([
    html.H2('Interaktive Tools'),
    # hier können Sie interaktive Tools hinzufügen, die es Benutzern ermöglichen, eigene Prognosen zu erstellen und die Leistung von Aktien auf der Grundlage von Daten und historischen Trends zu analysieren
])

# Layout definieren
app.layout = html.Div([
    market_data,
    stock_list,
    performance_charts,
    prediction_charts,
    algorithms,
    risk_analysis,
    notifications,
    interactive_tools
])

# CSS-Stylesheet


if __name__ == '__main__':
    app.run_server(debug=True)

