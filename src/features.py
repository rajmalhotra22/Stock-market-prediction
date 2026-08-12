import pandas as pd
import os

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# Load NIFTY 50 data
input_file = "data/nifty50_2021_2026.csv"
df = pd.read_csv(input_file, skiprows=[1, 2])

# Clean column names
df.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

# Convert numeric columns
numeric_columns = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# -----------------------------
# Technical Indicators
# -----------------------------

# Simple Moving Average
df["SMA_20"] = SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()

# Exponential Moving Average
df["EMA_20"] = EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

# Relative Strength Index
df["RSI"] = RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Histogram"] = macd.macd_diff()

# Bollinger Bands
bollinger = BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_Middle"] = bollinger.bollinger_mavg()
df["BB_Upper"] = bollinger.bollinger_hband()
df["BB_Lower"] = bollinger.bollinger_lband()


# -----------------------------
# Additional Features
# -----------------------------

# Daily percentage return
df["Daily_Return"] = df["Close"].pct_change()

# Volatility
df["Volatility_20"] = df["Daily_Return"].rolling(20).std()


# Remove rows containing NaN values
df.dropna(inplace=True)


# -----------------------------
# Save processed dataset
# -----------------------------

os.makedirs("data", exist_ok=True)

output_file = "data/nifty50_features_2021_2026.csv"

df.to_csv(output_file, index=False)


# -----------------------------
# Display results
# -----------------------------

print("Feature engineering completed successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFeatures:")
print(df.columns.tolist())

print("\nLast 5 rows:")
print(df.tail())