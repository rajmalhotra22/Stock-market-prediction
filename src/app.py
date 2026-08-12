import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
import subprocess
import sys
import re

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NIFTY 50 Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 NIFTY 50 Stock Market Prediction")
st.write("LSTM-based Stock Price Prediction System")

st.divider()

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "data/processed/lstm_model.keras"
    )


@st.cache_resource
def load_scaler():
    return joblib.load(
        "data/processed/scaler.pkl"
    )


model = load_model()
scaler = load_scaler()

# =========================================================
# LOAD TEST DATA
# =========================================================

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy"
)

# =========================================================
# PREDICTION
# =========================================================

predictions = model.predict(
    X_test,
    verbose=0
).flatten()

normalized_prediction = float(
    predictions[-1]
)

# =========================================================
# CONVERT NORMALIZED VALUE TO ORIGINAL PRICE
# =========================================================

# Close is the FIRST feature
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

historical_predicted_price = float(
    original_values[
        0,
        close_index
    ]
)

# =========================================================
# CONVERT TEST VALUES TO REAL NIFTY PRICES
# =========================================================

# Convert actual normalized values
actual_dummy = np.zeros(
    (len(y_test), 13)
)

actual_dummy[
    :,
    close_index
] = y_test

actual_prices = scaler.inverse_transform(
    actual_dummy
)[:, close_index]


# Convert predicted normalized values
predicted_dummy = np.zeros(
    (len(predictions), 13)
)

predicted_dummy[
    :,
    close_index
] = predictions

predicted_prices = scaler.inverse_transform(
    predicted_dummy
)[:, close_index]

# =========================================================
# SESSION STATE FOR FUTURE PREDICTION
# =========================================================

if "future_predicted_price" not in st.session_state:
    st.session_state.future_predicted_price = None

if "future_output" not in st.session_state:
    st.session_state.future_output = None

# =========================================================
# DASHBOARD TOP METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted NIFTY 50 Price",
        f"₹{historical_predicted_price:,.2f}"
    )

with col2:
    st.metric(
        "Model",
        "LSTM"
    )

with col3:
    st.metric(
        "Features",
        "13"
    )

st.divider()

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader("📊 Actual vs Predicted")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    actual_prices,
    label="Actual Price"
)

ax.plot(
    predicted_prices,
    label="Predicted Price"
)

ax.set_xlabel(
    "Test Samples"
)

ax.set_ylabel(
    "NIFTY 50 Price (₹)"
)

ax.set_title(
    "NIFTY 50 - Actual vs Predicted"
)

ax.legend()
ax.grid(True)

st.pyplot(fig)

plt.close(fig)

st.divider()

# =========================================================
# MODEL PERFORMANCE
# =========================================================

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📈 Model Performance")

# Normalized metrics
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

# Real price metrics
real_mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

real_rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

# =========================================================
# DISPLAY PERFORMANCE
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"₹{real_mae:,.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"₹{real_rmse:,.2f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

st.caption(
    f"Normalized MAE: {mae:.4f} | "
    f"Normalized RMSE: {rmse:.4f}"
)

st.divider()

# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("🤖 Model Information")

st.write(
    """
**Algorithm:** LSTM (Long Short-Term Memory)

**Dataset:** NIFTY 50

**Dataset Period:** 2021–2026

**Input Sequence:** 60 time steps

**Number of Features:** 13

**Data Scaling:** MinMaxScaler
"""
)

st.success(
    "Prediction system is running successfully!"
)

st.divider()

# =========================================================
# FUTURE PREDICTION
# =========================================================

st.subheader("🔮 Future NIFTY 50 Prediction")

if st.button(
    "🚀 Predict Future NIFTY 50 Price"
):

    try:

        result = subprocess.run(
            [
                sys.executable,
                "src/future_prediction.py"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            output = result.stdout

            # ---------------------------------------------
            # Extract predicted price
            # ---------------------------------------------

            match = re.search(
                r"Predicted Next Close Price:\s*₹?([\d,]+\.?\d*)",
                output
            )

            if match:

                predicted_text = match.group(1)

                future_price = float(
                    predicted_text.replace(
                        ",",
                        ""
                    )
                )

                st.session_state.future_predicted_price = (
                    future_price
                )

                st.session_state.future_output = output

                st.success(
                    "Future prediction generated successfully!"
                )

                st.code(output)

            else:

                st.error(
                    "Prediction generated, but predicted price could not be read."
                )

                st.code(output)

        else:

            st.error(
                "Future prediction failed."
            )

            st.code(
                result.stderr
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

st.divider()

# =========================================================
# PRICE COMPARISON
# =========================================================

st.subheader("📊 Price Comparison")

# =========================================================
# GET CURRENT NIFTY 50 PRICE
# =========================================================

try:

    nifty = yf.Ticker("^NSEI")

    latest_data = nifty.history(
        period="1d"
    )

    if not latest_data.empty:

        current_price = float(
            latest_data["Close"].iloc[-1]
        )

    else:

        current_price = 0.0

except Exception:

    current_price = 0.0

# =========================================================
# FUTURE PREDICTED PRICE
# =========================================================

future_price = (
    st.session_state.future_predicted_price
)

# =========================================================
# PRICE METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current NIFTY 50 Price",
        f"₹{current_price:,.2f}"
    )

with col2:

    if future_price is not None:

        st.metric(
            "Predicted NIFTY 50 Price",
            f"₹{future_price:,.2f}"
        )

    else:

        st.metric(
            "Predicted NIFTY 50 Price",
            "—"
        )

with col3:

    if (
        future_price is not None
        and current_price > 0
    ):

        change = (
            future_price
            - current_price
        )

        change_percent = (
            change
            / current_price
        ) * 100

        st.metric(
            "Expected Change",
            f"{change_percent:.2f}%"
        )

    else:

        st.metric(
            "Expected Change",
            "—"
        )

# =========================================================
# EXPECTED TREND
# =========================================================

if (
    future_price is not None
    and current_price > 0
):

    change = (
        future_price
        - current_price
    )

    if change > 0:

        st.success(
            "🟢 Expected Trend: UP"
        )

    elif change < 0:

        st.error(
            "🔴 Expected Trend: DOWN"
        )

    else:

        st.info(
            "⚪ Expected Trend: NEUTRAL"
        )

else:

    st.info(
        "👆 Click 'Predict Future NIFTY 50 Price' "
        "to generate the future prediction."
    )