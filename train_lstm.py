import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Load cleaned dataset
df = pd.read_csv("memory_cleaned.csv")

# Sort by experiment and time
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["experiment", "timestamp"])


# Features
features = [
    "memory_available_kb",
    "memory_free_kb",
    "memory_used_kb",
    "swap_used_kb",
    "page_faults_per_sec",
    "swap_in_per_sec",
    "swap_out_per_sec",
    "cpu_usage_percent",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "psi_some_avg10",
    "psi_some_avg60",
    "psi_full_avg10",
    "psi_full_avg60"
]

target = "pressure_next_5s"


# Store sequences
sequence_length = 10

X_sequences = []
y_sequences = []


# Create sequences separately for each experiment
for experiment, group in df.groupby("experiment"):

    group = group.sort_values("timestamp")

    X_data = group[features].values
    y_data = group[target].values

    # Scale features within each experiment
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)

    for i in range(sequence_length, len(group)):
        X_sequences.append(X_data[i-sequence_length:i])
        y_sequences.append(y_data[i])


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)


print("Sequence data shape:", X_sequences.shape)
print("Target shape:", y_sequences.shape)

print("\nTarget distribution:")
print(pd.Series(y_sequences).value_counts())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.20,
    random_state=42,
    stratify=y_sequences
)

print("\nTraining sequences:", len(X_train))
print("Testing sequences:", len(X_test))


# Build LSTM model
model = Sequential([
    LSTM(32, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining LSTM...")

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=1
)

print("\nLSTM training completed!")


# Predictions
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)


# Results
print("\n======================================")
print("             LSTM RESULTS")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nStep 3D complete!")