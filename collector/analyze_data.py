import csv

file_name = "data/memory_dataset_labeled.csv"

with open(file_name, "r", newline="") as f:
    rows = list(csv.DictReader(f))

normal = [r for r in rows if r["pressure_next_5s"] == "0"]
pressure = [r for r in rows if r["pressure_next_5s"] == "1"]

def values(data, column):
    return [float(r[column]) for r in data]

print("\n===== DATASET SUMMARY =====")

print("Total samples:", len(rows))
print("Normal samples:", len(normal))
print("Pressure onset samples:", len(pressure))

print("\n===== NORMAL VS PRESSURE =====")

columns = [
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
    "psi_full_avg10"
]

for column in columns:
    normal_values = values(normal, column)
    pressure_values = values(pressure, column)

    print("\n", column)
    print("  Normal average:  ", round(sum(normal_values) / len(normal_values), 2))
    print("  Pressure average:", round(sum(pressure_values) / len(pressure_values), 2))

print("\n===== PRESSURE ONSET RANGE =====")

for column in columns:
    pressure_values = values(pressure, column)

    print(
        column,
        "| min:", round(min(pressure_values), 2),
        "| max:", round(max(pressure_values), 2)
    )
