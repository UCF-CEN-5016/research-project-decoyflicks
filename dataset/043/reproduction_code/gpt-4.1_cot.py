# Directory structure is assumed:
# examples/mms/commons.py
# examples/mms/tts/infer.py

# Content of commons.py (examples/mms/commons.py)
# This is a dummy module to simulate the real commons module.
def dummy_function():
    print("Dummy function from commons")

# Content of infer.py (examples/mms/tts/infer.py)
try:
    import commons
    commons.dummy_function()
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")