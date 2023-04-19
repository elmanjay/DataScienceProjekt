import pandas as pd
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

msft = yf.Ticker("msft")
df= msft.history(period="max")
df.reset_index(inplace= True)
print(df["Open"].max())