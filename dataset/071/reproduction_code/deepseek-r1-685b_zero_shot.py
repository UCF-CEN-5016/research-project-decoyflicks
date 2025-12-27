from transformers.deepspeed import is_deepspeed_zero3_enabled

def main():
    print(f"DeepSpeed Zero3 enabled: {is_deepspeed_zero3_enabled()}")

if __name__ == "__main__":
    main()