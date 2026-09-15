import csv
import matplotlib.pyplot as plt

file_name = "data/memory_dataset_labeled.csv"

with open(file_name, "r", newline="") as f:
    rows = list(csv.DictReader(f))

pressure_rows = [r for r in rows if r["pressure_next_5s"] == "1"]

memory = [
    float(r["memory_available_kb"]) / 1024
    for r in pressure_rows
]

print("Pressure onset samples:", len(memory))
print("Average available memory:", round(sum(memory) / len(memory), 2), "MB")

plt.plot(range(1, len(memory) + 1), memory, marker="o")

plt.xlabel("Pressure-onset sample")
plt.ylabel("Available memory (MB)")
plt.title("Available Memory Before Predicted Memory Pressure")

plt.tight_layout()
plt.show()

