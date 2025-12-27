import os

# Simulate fairseq installation and directory structure
fairseq_root = "fairseq"
examples_mms_tts = os.path.join(fairseq_root, "examples", "mms", "tts")

# Create minimal directory structure (simulated)
os.makedirs(examples_mms_tts, exist_ok=True)

def import_commons():
    try:
        from commons import text_to_sequence  # This will fail
        print("Successfully imported commons")
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        print("This reproduces the bug - 'commons' module not found")

if __name__ == "__main__":
    # Change to the TTS directory (as user would do)
    os.chdir(examples_mms_tts)

    # Simulate the infer.py script trying to import commons
    import_commons()

    # Cleanup (optional)
    os.chdir("../../../../")