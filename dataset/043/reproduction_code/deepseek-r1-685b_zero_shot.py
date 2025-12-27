import sys
from pathlib import Path

# Simulate fairseq repo structure
root = Path("fairseq")
(root / "examples/mms/tts").mkdir(parents=True, exist_ok=True)
(root / "fairseq").mkdir(exist_ok=True)

# Create dummy infer.py
infer_content = """
from commons import some_function

def main():
    some_function()

if __name__ == "__main__":
    main()
"""
(root / "examples/mms/tts/infer.py").write_text(infer_content)

# Try to run infer.py
sys.path.insert(0, str(root))
import examples.mms.tts.infer