import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = tf.keras.models.load_model(
    "data/processed/lstm_model.keras"
)


# =========================================================
# LOAD SCALER
# =========================================================

scaler = joblib.load(
    "data/processed/scaler.pkl"
)


# =========================================================
# LOAD FEATURE DATA
# =========================================================

df = pd.read_csv(
    "data/nifty50_features_2021_2026.csv"
)


# =========================================================
# 13 FEATURES USED DURING TRAINING
# =========================================================

features = [
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "Daily_Return",
    "Volatility_20"
]


# =========================================================
# PREPARE DATA
# =========================================================

data = df[features].copy()

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna().reset_index(
    drop=True
)


# =========================================================
# TAKE LATEST 60 DAYS
# =========================================================

sequence_length = 60

latest_data = data.tail(
    sequence_length
)


if len(latest_data) < sequence_length:

    raise ValueError(
        "Not enough data available for prediction."
    )


# =========================================================
# SCALE DATA
# =========================================================

scaled_latest_data = scaler.transform(
    latest_data
)


# =========================================================
# CREATE LSTM INPUT
# =========================================================

X_future = np.array(
    [scaled_latest_data]
)


# Expected shape:
# (1, 60, 13)

print(
    "Input shape:",
    X_future.shape
)


# =========================================================
# PREDICT NEXT CLOSE PRICE
# =========================================================

prediction = model(
    X_future,
    training=False
).numpy()


normalized_prediction = float(
    prediction[0][0]
)


# =========================================================
# CONVERT NORMALIZED VALUE TO REAL PRICE
# =========================================================

# Close is the first feature
close_index = 0


dummy = np.zeros(
    (1, 13)
)


dummy[
    0,
    close_index
] = normalized_prediction


original_values = scaler.inverse_transform(
    dummy
)


predicted_price = float(
    original_values[
        0,
        close_index
    ]
)


# =========================================================
# GET CURRENT/LATEST NIFTY 50 PRICE
# =========================================================

try:

    nifty = yf.Ticker("^NSEI")

    latest_market_data = nifty.history(
        period="1d"
    )

    if latest_market_data.empty:

        raise ValueError(
            "Unable to fetch latest NIFTY 50 market price."
        )

    current_price = float(
        latest_market_data["Close"].iloc[-1]
    )

except Exception as e:

    print(
        "Warning: Unable to fetch live NIFTY 50 price."
    )

    print(
        "Using latest dataset price instead."
    )

    current_price = float(
        latest_data["Close"].iloc[-1]
    )


# =========================================================
# EXPECTED CHANGE
# =========================================================

change = (
    predicted_price -
    current_price
)


change_percent = (
    change /
    current_price
) * 100


# =========================================================
# EXPECTED TREND
# =========================================================

if change > 0:

    trend = "UP"

else:

    trend = "DOWN"


# =========================================================
# DISPLAY RESULT
# =========================================================

print(
    "\n======================================"
)

print(
    "       NIFTY 50 FUTURE PREDICTION"
)

print(
    "======================================"
)


print(
    f"Latest NIFTY 50 Close: ₹{current_price:.2f}"
)


print(
    f"Normalized prediction: "
    f"{normalized_prediction:.6f}"
)


print(
    f"Predicted Next Close Price: "
    f"₹{predicted_price:.2f}"
)


print(
    f"Expected Change: "
    f"{change_percent:.2f}%"
)


print(
    f"Expected Trend: {trend}"
)


print(
    "\nPrediction generated successfully!"
)