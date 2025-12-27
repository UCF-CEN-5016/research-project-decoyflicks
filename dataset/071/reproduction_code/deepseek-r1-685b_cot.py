# Reproduction code for "No module named 'transformers.deepspeed'" error
import transformers

def test_deepspeed_import():
    try:
        # This is the import that fails in the original script
        from transformers.deepspeed import HfDeepSpeedConfig
        print("Successfully imported transformers.deepspeed")
    except ImportError as e:
        print(f"Import failed: {e}")

if __name__ == "__main__":
    print(f"Transformers version: {transformers.__version__}")
    test_deepspeed_import()