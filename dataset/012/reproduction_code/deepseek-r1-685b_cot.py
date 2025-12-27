# Reproduction of AttributeError in tf-keras optimizer legacy module
import tf_keras
from tf_keras.optimizers import legacy

try:
    # This is what the EMA optimizer in the original code is trying to do
    optimizer_base = legacy.Optimizer
except AttributeError as e:
    print(f"Error reproduced: {e}")
    print("Available attributes in legacy module:", dir(legacy))

# Additional context to show the expected vs actual state
print("\nDebug information:")
print(f"tf_keras version: {tf_keras.__version__}")
print("Expected: legacy module should contain Optimizer class")
print("Actual: legacy module contains:", [attr for attr in dir(legacy) if not attr.startswith('_')])