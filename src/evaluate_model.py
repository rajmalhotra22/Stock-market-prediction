import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model
model = tf.keras.models.load_model(
    "data/processed/lstm_model.keras"
)

# Load test data
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Predictions
predictions = model.predict(X_test).flatten()
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n==============================")
print("     MODEL PERFORMANCE")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot
plt.figure(figsize=(12, 6))

plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("NIFTY 50 - Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("data/processed/actual_vs_predicted.png", dpi=300)

plt.show()

print("\nGraph generated successfully!")
print("Saved as: data/processed/actual_vs_predicted.png")