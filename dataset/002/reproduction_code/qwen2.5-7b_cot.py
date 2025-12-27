# repro.py
import tensorflow as tf

try:
    from tensorflow_model_optimization.python.core.quantization.keras import quantize
    print("Successfully imported quantize from tensorflow_model_optimization")
except ImportError:
    print("Failed to import quantize from tensorflow_model_optimization")