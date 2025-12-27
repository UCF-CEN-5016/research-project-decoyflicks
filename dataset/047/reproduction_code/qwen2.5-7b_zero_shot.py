import sys
import threading
import time
import random

def process_audio(file):
    time.sleep(random.random())  # Simulate processing delay
    with open('output.log', 'a') as f:
        f.write(f"Processed {file}\n")

if __name__ == '__main__':
    files = sys.argv[1:]
    threads = []
    
    for file in files:
        t = threading.Thread(target=process_audio, args=(file,))
        threads.append(t)
        t.start()
    
    for thread in threads:
        thread.join()