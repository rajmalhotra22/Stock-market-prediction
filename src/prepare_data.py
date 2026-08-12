import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/nifty50_features_2021_2026.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


# ==============================
# 2. Select Features
# ==============================

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

data = df[features].copy()

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna().reset_index(drop=True)


# ==============================
# 3. Create 60-Day Sequences
# ==============================

sequence_length = 60

total_rows = len(data)

# Raw-data boundaries
train_row_end = int(total_rows * 0.70)
validation_row_end = int(total_rows * 0.85)


# ==============================
# 4. Fit Scaler ONLY on Training Data
# ==============================

scaler = MinMaxScaler()

scaler.fit(data.iloc[:train_row_end])

scaled_data = scaler.transform(data)


# Save scaler
os.makedirs("data/processed", exist_ok=True)

joblib.dump(
    scaler,
    "data/processed/scaler.pkl"
)


# ==============================
# 5. Create Sequences
# ==============================

X = []
y = []
target_indices = []

close_index = features.index("Close")

for i in range(sequence_length, len(scaled_data)):

    X.append(
        scaled_data[i - sequence_length:i]
    )

    y.append(
        scaled_data[i, close_index]
    )

    target_indices.append(i)


X = np.array(X)
y = np.array(y)
target_indices = np.array(target_indices)


# ==============================
# 6. Chronological Split
# ==============================

train_mask = target_indices < train_row_end

validation_mask = (
    (target_indices >= train_row_end) &
    (target_indices < validation_row_end)
)

test_mask = target_indices >= validation_row_end


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# ==============================
# 7. Save Data
# ==============================

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)

np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)


# ==============================
# 8. Display
# ==============================

print("\nPreprocessing completed successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nScaler saved successfully!")