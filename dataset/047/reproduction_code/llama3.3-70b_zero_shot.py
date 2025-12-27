import threading
import time
import random

class MMSInfer:
    def __init__(self):
        self.lock = threading.Lock()
        self.audio_files = ["audio1.wav", "audio2.wav", "audio3.wav", "audio4.wav", "audio5.wav", "audio6.wav", "audio7.wav", "audio8.wav", "audio9.wav", "audio10.wav"]
        self.outputs = {}

    def infer(self, audio_file):
        time.sleep(random.uniform(0, 1))  # Simulate inference time
        output = f"Inferred output for {audio_file}"
        with self.lock:
            self.outputs[audio_file] = output

    def run_inference(self):
        threads = []
        for audio_file in self.audio_files:
            thread = threading.Thread(target=self.infer, args=(audio_file,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for audio_file in self.audio_files:
            print(f"Input: {audio_file}")
            print(f"Output: {self.outputs[audio_file]}")

mms_infer = MMSInfer()
mms_infer.run_inference()