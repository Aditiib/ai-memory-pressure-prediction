import time
import joblib
import pandas as pd


model = joblib.load("memory_model.pkl")


def get_value(file, key):
    with open(file, "r") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    return 0


def get_cpu_times():
    with open("/proc/stat", "r") as f:
        parts = f.readline().split()

    values = list(map(int, parts[1:]))

    total = sum(values)
    idle = values[3] + values[4]

    return total, idle


def get_disk():
    read_bytes = 0
    write_bytes = 0

    with open("/proc/diskstats", "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) >= 14:
                name = parts[2]

                if name.startswith(("sd", "vd", "nvme")):
                    read_bytes += int(parts[5]) * 512
                    write_bytes += int(parts[9]) * 512

    return read_bytes, write_bytes


def get_psi():
    with open("/proc/pressure/memory", "r") as f:
        lines = f.readlines()

    some = lines[0].split()
    full = lines[1].split()

    some10 = float(some[1].split("=")[1])
    some60 = float(some[2].split("=")[1])

    full10 = float(full[1].split("=")[1])
    full60 = float(full[2].split("=")[1])

    return some10, some60, full10, full60


previous_cpu_total, previous_cpu_idle = get_cpu_times()
previous_disk_read, previous_disk_write = get_disk()

previous_page_faults = get_value("/proc/vmstat", "pgfault")
previous_swap_in = get_value("/proc/vmstat", "pswpin")
previous_swap_out = get_value("/proc/vmstat", "pswpout")

previous_time = time.monotonic()


print("==============================================")
print("     LIVE MEMORY PRESSURE PREDICTION")
print("==============================================")
print("Monitoring Ubuntu system...")
print("Press Ctrl+C to stop.")
print()


while True:

    time.sleep(1)

    now = time.monotonic()
    elapsed = now - previous_time

    memory_total = get_value("/proc/meminfo", "MemTotal:")
    memory_available = get_value("/proc/meminfo", "MemAvailable:")
    memory_free = get_value("/proc/meminfo", "MemFree:")

    swap_total = get_value("/proc/meminfo", "SwapTotal:")
    swap_free = get_value("/proc/meminfo", "SwapFree:")
    swap_used = swap_total - swap_free

    memory_used = memory_total - memory_available

    page_faults = get_value("/proc/vmstat", "pgfault")
    swap_in = get_value("/proc/vmstat", "pswpin")
    swap_out = get_value("/proc/vmstat", "pswpout")

    cpu_total, cpu_idle = get_cpu_times()

    disk_read, disk_write = get_disk()

    cpu_delta = cpu_total - previous_cpu_total
    idle_delta = cpu_idle - previous_cpu_idle

    if cpu_delta > 0:
        cpu_usage = 100 * (1 - idle_delta / cpu_delta)
    else:
        cpu_usage = 0

    page_faults_per_sec = max(
        0, (page_faults - previous_page_faults) / elapsed
    )

    swap_in_per_sec = max(
        0, (swap_in - previous_swap_in) / elapsed
    )

    swap_out_per_sec = max(
        0, (swap_out - previous_swap_out) / elapsed
    )

    disk_read_per_sec = max(
        0, (disk_read - previous_disk_read) / elapsed
    )

    disk_write_per_sec = max(
        0, (disk_write - previous_disk_write) / elapsed
    )

    some10, some60, full10, full60 = get_psi()

    input_data = pd.DataFrame([[
        memory_available,
        memory_free,
        memory_used,
        swap_used,
        page_faults_per_sec,
        swap_in_per_sec,
        swap_out_per_sec,
        cpu_usage,
        disk_read_per_sec,
        disk_write_per_sec,
        some10,
        some60,
        full10,
        full60
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

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    print("----------------------------------------------")

    print(
        f"Available Memory : {memory_available / 1024:.2f} MB"
    )

    print(
        f"Used Memory      : {memory_used / 1024:.2f} MB"
    )

    print(
        f"Swap Used        : {swap_used / 1024:.2f} MB"
    )

    print(
        f"Page Faults      : {page_faults_per_sec:.2f}/sec"
    )

    print(
        f"CPU Usage        : {cpu_usage:.2f}%"
    )

    if prediction == 1:
        print()
        print("⚠️  PRESSURE EXPECTED")
        print(
            f"Pressure Probability: {probability * 100:.2f}%"
        )
    else:
        print()
        print("✅ NORMAL")
        print(
            f"Pressure Probability: {probability * 100:.2f}%"
        )

    previous_cpu_total = cpu_total
    previous_cpu_idle = cpu_idle

    previous_disk_read = disk_read
    previous_disk_write = disk_write

    previous_page_faults = page_faults
    previous_swap_in = swap_in
    previous_swap_out = swap_out

    previous_time = now
