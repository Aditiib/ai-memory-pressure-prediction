import matplotlib.pyplot as plt

# ============================================================
# MODEL RESULTS
# ============================================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "LSTM"
]

accuracy = [86.41, 99.03, 99.03, 97.72]

precision = [11.11, 80.00, 66.67, 0.00]

recall = [100.00, 57.14, 85.71, 0.00]

f1_score = [20.00, 66.67, 75.00, 0.00]

# ============================================================
# GRAPH 1: ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

# Show values on top of bars
for i, value in enumerate(accuracy):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

# ============================================================
# GRAPH 2: PRECISION, RECALL AND F1 SCORE
# ============================================================

x = range(len(models))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width for i in x],
    precision,
    width=width,
    label="Precision"
)

plt.bar(
    x,
    recall,
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    f1_score,
    width=width,
    label="F1 Score"
)

plt.title("Precision, Recall and F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("Score (%)")

plt.ylim(0, 110)

plt.xticks(list(x), models, rotation=15)

plt.legend()

plt.tight_layout()

plt.savefig("precision_recall_f1.png", dpi=300)
plt.show()

# ============================================================
# GRAPH 3: CLASS DISTRIBUTION
# ============================================================

classes = [
    "Normal (0)",
    "Pressure (1)"
]

counts = [
    2021,
    37
]

plt.figure(figsize=(8, 6))

plt.bar(classes, counts)

plt.title("Pressure Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Observations")

# Show values on top
for i, value in enumerate(counts):
    plt.text(i, value + 20, str(value), ha="center")

plt.tight_layout()

plt.savefig("class_distribution.png", dpi=300)
plt.show()

print("====================================")
print("All graphs created successfully!")
print("====================================")
print("1. accuracy_comparison.png")
print("2. precision_recall_f1.png")
print("3. class_distribution.png")