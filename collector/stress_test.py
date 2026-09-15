import time

blocks = []
block_size = 50 * 1024 * 1024

try:
    print("Starting memory pressure test...")

    for i in range(75):
        block = bytearray(block_size)

        for j in range(0, len(block), 4096):
            block[j] = 1

        blocks.append(block)

        print("Allocated:", (i + 1) * 50, "MB")
        time.sleep(1)

    print("Memory allocation complete.")
    print("Starting memory churn...")

    for cycle in range(30):
        for i in range(0, len(blocks), 2):
            blocks[i] = bytearray(block_size)

            for j in range(0, len(blocks[i]), 4096):
                blocks[i][j] = 1

        print("Churn cycle:", cycle + 1)
        time.sleep(1)

    print("Holding memory...")
    time.sleep(30)

finally:
    blocks.clear()
    print("Memory released.")
