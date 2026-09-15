import joblib
import pandas as pd

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load("memory_model.pkl")

print("====================================")
print("AI Memory Pressure Prediction")
print("====================================")
print("Enter the current system values.")
print()

# ============================================================
# GET INPUT VALUES
# ============================================================

memory_available_kb = float(input("Memory available (KB): "))
memory_free_kb = float(input("Memory free (KB): "))
memory_used_kb = float(input("Memory used (KB): "))
swap_used_kb = float(input("Swap used (KB): "))
page_faults_per_sec = float(input("Page faults per second: "))
swap_in_per_sec = float(input("Swap in per second: "))
swap_out_per_sec = float(input("Swap out per second: "))
cpu_usage_percent = float(input("CPU usage (%): "))
disk_read_bytes_per_sec = float(input("Disk read (bytes/sec): "))
disk_write_bytes_per_sec = float(input("Disk write (bytes/sec): "))
psi_some_avg10 = float(input("PSI some avg10: "))
psi_some_avg60 = float(input("PSI some avg60: "))
psi_full_avg10 = float(input("PSI full avg10: "))
psi_full_avg60 = float(input("PSI full avg60: "))

# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame([[
    memory_available_kb,
    memory_free_kb,
    memory_used_kb,
    swap_used_kb,
    page_faults_per_sec,
    swap_in_per_sec,
    swap_out_per_sec,
    cpu_usage_percent,
    disk_read_bytes_per_sec,
    disk_write_bytes_per_sec,
    psi_some_avg10,
    psi_some_avg60,
    psi_full_avg10,
    psi_full_avg60
]], columns=[
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
])

# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]

# Probability of pressure
probability = model.predict_proba(input_data)[0][1]

# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("====================================")
print("Prediction Result")
print("====================================")

if prediction == 1:
    print("⚠️ PRESSURE EXPECTED")
    print("Memory pressure may occur within the next 5 seconds.")
else:
    print("✅ NORMAL")
    print("No memory pressure predicted within the next 5 seconds.")

print(f"Pressure probability: {probability * 100:.2f}%")
print("====================================")