import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load cleaned dataset
df = pd.read_csv("memory_cleaned.csv")

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

# Target
target = "pressure_next_5s"

X = df[features]
y = df[target]

# Same split used during model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "memory_model.pkl")

print("====================================")
print("Random Forest model saved!")
print("====================================")
print("File: memory_model.pkl")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))