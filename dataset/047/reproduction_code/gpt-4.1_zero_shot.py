import threading
import time
import random

def infer(audio_path):
    time.sleep(random.uniform(0.1, 0.5))
    print(f"Input: {audio_path}")
    print(f"Output: transcript of {audio_path}")

audios = [f"1089-134686-000{i}.wav" for i in range(10)]

threads = []
for audio in audios:
    t = threading.Thread(target=infer, args=(audio,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()