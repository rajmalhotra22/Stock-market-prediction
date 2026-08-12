import yfinance as yf
import os

ticker = "^NSEI"

# Download NIFTY 50 data from 2021 to latest available date
data = yf.download(
    ticker,
    start="2021-01-01",
    auto_adjust=False
)

# Create data folder
os.makedirs("data", exist_ok=True)

# Save data
data.to_csv("data/nifty50_2021_2026.csv")

print("NIFTY 50 data downloaded successfully!")
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())