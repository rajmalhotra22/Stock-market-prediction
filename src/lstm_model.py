import numpy as np
import tensorflow as tf


# =========================================================
# 1. LOAD PROCESSED DATA
# =========================================================

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")


print("Data loaded successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =========================================================
# 2. BUILD LSTM MODEL
# =========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60, 13)),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(32),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(1)
])


# =========================================================
# 3. COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# =========================================================
# 4. CALLBACKS
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "data/processed/best_lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# =========================================================
# 5. TRAIN MODEL
# =========================================================

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# =========================================================
# 6. EVALUATE MODEL
# =========================================================

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\nTraining completed successfully!")

print("Test Loss:", test_loss)
print("Test MAE:", test_mae)


# =========================================================
# 7. SAVE FINAL MODEL
# =========================================================

model.save(
    "data/processed/lstm_model.keras"
)


print("\nLSTM model saved successfully!")