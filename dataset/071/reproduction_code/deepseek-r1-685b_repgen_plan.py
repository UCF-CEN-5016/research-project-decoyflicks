import transformers

# Attempt to import deepspeed module from transformers
try:
    import transformers.deepspeed
    print("DeepSpeed module found in transformers")
except ImportError as e:
    print(f"Error: {e}")
    print("This occurs when the installed transformers package doesn't include deepspeed integration")

# Verify package versions
print(f"transformers version: {transformers.__version__}")
print(f"deepspeed version: {getattr(transformers, '__deepspeed_version__', 'Not available')}")