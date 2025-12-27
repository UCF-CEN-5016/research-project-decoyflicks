import sys
from transformers import AutoModelForCausalLM

def main():
    try:
        from transformers.deepspeed import HfDeepSpeedConfig
    except ImportError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()