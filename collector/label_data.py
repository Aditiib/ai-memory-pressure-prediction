import csv
files = [
    ("test_1", "data/memory_test_1.csv"),
    ("test_2", "data/memory_test_2.csv"),
    ("test_3_pressure", "data/memory_test_3_pressure.csv"),
    ("test_4", "data/memory_test_4.csv"),
    ("test_5", "data/memory_test_5.csv"),
    ("test_6", "data/memory_test_6.csv"),
    ("test_7", "data/memory_test_7.csv"),
    ("test_8", "data/memory_test_8.csv"),
    ("test_9", "data/memory_test_9.csv")
]
output_file = "data/memory_dataset_labeled.csv"

all_rows = []

for test_name, file_name in files:

    with open(file_name, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    pressure = []

    for row in rows:
        swap_out = float(row["swap_out_per_sec"])
        psi_some = float(row["psi_some_avg10"])
        psi_full = float(row["psi_full_avg10"])

        if swap_out > 0 or psi_some > 0 or psi_full > 0:
            pressure.append(1)
        else:
            pressure.append(0)

    pressure_start = []

    for i in range(len(rows)):
        if pressure[i] == 1 and (i == 0 or pressure[i - 1] == 0):
            pressure_start.append(i)

    labels = [0] * len(rows)

    for start in pressure_start:
        for i in range(max(0, start - 5), start):
            labels[i] = 1

    for i in range(len(rows) - 5):
        current = rows[i].copy()

        current["experiment"] = test_name
        current["pressure_next_5s"] = labels[i]

        all_rows.append(current)


fieldnames = list(all_rows[0].keys())

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(all_rows)


ones = sum(int(row["pressure_next_5s"]) for row in all_rows)

print("Created:", output_file)
print("Total rows:", len(all_rows))
print("Pressure onset prediction = 1:", ones)
print("Normal = 0:", len(all_rows) - ones)
