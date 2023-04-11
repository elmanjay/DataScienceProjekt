import pandas as pd
import yfinance as yf

# Erstellen Sie ein Objekt für das zu beobachtende Unternehmen
msft = yf.Ticker("MSFT")
df= msft.history(period="max")
print(df.head())
