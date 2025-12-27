import transformers

# Attempt to import deepspeed module from transformers
try:
    from transformers.deepspeed import is_deepspeed_available
    print("DeepSpeed module found in transformers")
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("This occurs when the installed transformers package doesn't include deepspeed integration")

# Verify package versions
print(f"transformers version: {transformers.__version__}")
print(f"deepspeed version: {getattr(transformers, '__deepspeed_version__', 'Not available')}")

# Expected output when bug occurs:
# Error: No module named 'transformers.deepspeed'
# transformers version: [current version]
# deepspeed version: Not available