import sys
sys.setrecursionlimit(10000)

from tf_keras.optimizers.legacy import ExponentialMovingAverage

# Attempting to create an EMA without any parameters triggers the error
try:
    ema = ExponentialMovingAverage()
except Exception as e:
    print(f"Error: {e}")