import numpy as np
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model(
    "data/processed/lstm_model.keras"
)

# Load test data
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Make predictions
predictions = model.predict(X_test)

print("\nPrediction completed successfully!")

print("\nActual values:")
print(y_test[:10])

print("\nPredicted values:")
print(predictions[:10].flatten())