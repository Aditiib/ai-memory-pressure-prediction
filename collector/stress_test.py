import time

blocks = []
block_size = 10 * 1024 * 1024

try:
    print("Starting gradual memory pressure test...")

    for i in range(340):
        block = bytearray(block_size)

        for j in range(0, len(block), 4096):
            block[j] = 1

        blocks.append(block)

        print("Allocated:", (i + 1) * 10, "MB")
        time.sleep(0.4)

    print()
    print("Memory allocation complete.")
    print("Holding memory...")
    time.sleep(20)

finally:
    blocks.clear()
    print("Memory released.")
