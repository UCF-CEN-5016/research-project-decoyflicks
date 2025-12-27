# Step 1: Minimal import to trigger the error
try:
    from transformers import deepspeed
    print("Import successful: transformers.deepspeed found")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")

# Step 2: Check transformers version
import transformers
print(f"transformers version: {transformers.__version__}")

# Step 3: Check if deepspeed package is installed
try:
    import deepspeed
    print("DeepSpeed package is installed")
except ModuleNotFoundError:
    print("DeepSpeed package is NOT installed")