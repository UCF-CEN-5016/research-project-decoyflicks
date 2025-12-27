import torch
import numpy as np
import threading
import time

# Simulated shared data structure
shared_data = []

def worker(index):
    # Simulate processing and modify shared data
    if index == 0:
        shape_a = [4,2,4]
    else:
        shape_b = [4,1,4]
    shared_data.append(shape_a)
    shared_data.append(shape_b)

# Start workers
print("Starting threads...")
threads = []
for i in range(5):
    if i == 0:
        print(f"Worker {i} (shape A)")
    else:
        print(f"Worker {i} (shape B)")
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all workers to finish
for t in threads:
    t.join()

# Check the final state of shared data
print("\nFinal shared data state:")
print("Worker 0 added:", shared_data[0] if len(shared_data) else "None")
if len(shared_data) > 1 and (shared_data[0].shape[1] != shared_data[1].shape[1]):
    print(f"Shapes mismatch: {shared_data[0].shape} vs. {shared_data[1].shape}")