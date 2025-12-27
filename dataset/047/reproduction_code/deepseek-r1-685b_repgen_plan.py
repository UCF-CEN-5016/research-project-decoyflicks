import time
import os

# Mock model and inference function
class MockMMSModel:
    def __init__(self):
        self.results = {
            "audio1.wav": "result for audio1",
            "audio2.wav": "result for audio2", 
            "audio3.wav": "result for audio3"
        }
    
    def transcribe(self, audio_path):
        # Simulate async processing delay that causes order mismatch
        time.sleep(len(audio_path) * 0.01)  # Longer paths take longer
        return self.results[os.path.basename(audio_path)]

def test_mms_order():
    # Setup minimal test case
    audio_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav"
    ]

    model = MockMMSModel()
    
    print("Input order:")
    for f in audio_files:
        print(f)
    
    print("\nOutput order (incorrect):")
    outputs = [model.transcribe(f) for f in audio_files]
    
    for f, out in zip(audio_files, outputs):
        print(f"{f} -> {out}")
    
    # Verify order mismatch
    assert [out.split()[-1] for out in outputs] != ["audio1", "audio2", "audio3"]

test_mms_order()