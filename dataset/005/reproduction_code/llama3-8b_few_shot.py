import tensorflow as tf

# Minimal reproduction code that reproduces the error
from tf_keras.src.dtensor import lazy_variable

print("This should not run due to the import error")