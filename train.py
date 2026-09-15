import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==========================================
# 1. Load cleaned dataset
# ==========================================

df = pd.read_csv("memory_cleaned.csv")

# ==========================================
# 2. Select input features
# ==========================================

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

X = df[features]

# Target: pressure within next 5 seconds
y = df["pressure_next_5s"]

# ==========================================
# 3. Split data into training and testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ==========================================
# 4. Scale the features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. Create Logistic Regression model
# ==========================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# ==========================================
# 6. Train the model
# ==========================================

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")

# ==========================================
# 7. Make predictions
# ==========================================

y_pred = model.predict(X_test_scaled)

# Probability of pressure = 1
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# 8. Calculate evaluation metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# ==========================================
# 9. Display results
# ==========================================

print("\n======================================")
print("      LOGISTIC REGRESSION RESULTS")
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

print("\nStep 3A complete!")